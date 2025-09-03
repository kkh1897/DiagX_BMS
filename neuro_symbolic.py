"""
Neuro-symbolic multi-class classifier with interpretable rules.
- Adds compact band logic (disjoint & covering) with k_max=5 feature limit.
- Removes nonessential plotting.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler


# =========================================================================== #
# CORE ARCHITECTURAL COMPONENTS
# =========================================================================== #

class InterpretableRule(nn.Module):
    """
    Represents a single, learnable logical rule (e.g., "feature > threshold").

    This module acts as a differentiable sigmoid gate over a single input feature,
    learning an optimal threshold, direction (> or <=), and steepness.
    """
    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.rand(1))
        self.steepness = nn.Parameter(torch.ones(1))
        self.direction_weight = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the rule, returning a value between 0 and 1."""
        direction = torch.tanh(self.direction_weight)
        # Use softplus for a smooth, non-negative steepness parameter
        k = F.softplus(self.steepness)
        return torch.sigmoid(k * direction * (x - self.threshold))

    @torch.no_grad()
    def describe(self, feature_name: str, scaler: RobustScaler, feature_idx: int) -> str:
        """Generates a human-readable description of the learned rule."""
        t_scaled = self.threshold.item()
        
        # Create a dummy array to inverse-transform only the relevant feature's threshold
        dummy_array = np.zeros((1, scaler.n_features_in_))
        dummy_array[0, feature_idx] = t_scaled
        t_original = scaler.inverse_transform(dummy_array)[0, feature_idx]
        
        direction_val = torch.tanh(self.direction_weight).item()
        op = ">" if direction_val > 0 else "<="
        return f"{feature_name} {op} {t_original:.4f}"


class AdvancedLoss(nn.Module):
    """
    A composite loss function for training the neuro-symbolic model.

    It combines multiple objectives using uncertainty-aware weighting to balance
    their contributions automatically. It also includes regularization terms to
    encourage rule diversity and prediction decisiveness.
    """
    def __init__(self, num_classes: int, config: object, eps: float = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        # Key class indices and loss hyperparameters from a config object
        self.normal_class_idx = int(config.NORMAL_CLASS_IDX)
        self.fault_class_idx = int(config.FAULT_CLASS_IDX)
        self.beta = float(config.LOSS_BETA)  # For F-beta score
        self.gamma = float(config.LOSS_FOCAL_GAMMA) # For Focal Loss
        self.ortho_weight = float(config.LOSS_ORTHO_WEIGHT)

        # Learnable parameters for uncertainty weighting (log variances)
        self.log_var_fbeta = nn.Parameter(torch.zeros(1))
        self.log_var_fpr = nn.Parameter(torch.zeros(1))
        self.log_var_ce = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_rule_vectors: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Calculates the total loss."""
        probs = F.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float()

        # --- Base Loss Calculations ---
        loss_fbeta = self._fbeta_loss(probs, one_hot_targets)
        loss_fpr = self._fpr_loss(probs, one_hot_targets)
        ce_loss = F.cross_entropy(logits, targets)
        
        # Apply focal scaling to focus on hard examples
        pt = torch.exp(-ce_loss)
        focal_ce = ((1 - pt) ** self.gamma * ce_loss).mean()

        # --- Uncertainty-Weighted Combination ---
        weighted_loss_fbeta = torch.exp(-self.log_var_fbeta) * loss_fbeta + self.log_var_fbeta
        weighted_loss_fpr = torch.exp(-self.log_var_fpr) * loss_fpr + self.log_var_fpr
        weighted_loss_ce = torch.exp(-self.log_var_ce) * focal_ce + self.log_var_ce
        
        total_loss = weighted_loss_fbeta + weighted_loss_fpr + weighted_loss_ce

        # --- Regularization ---
        if class_rule_vectors:
            total_loss += self.ortho_weight * self._orthogonal_loss(class_rule_vectors)
        
        return total_loss

    def _fbeta_loss(self, probs: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        """Calculates F-beta loss, prioritizing recall for the fault class."""
        tp = (probs * one_hot).sum(dim=0)
        fp = (probs * (1 - one_hot)).sum(dim=0)
        fn = ((1 - probs) * one_hot).sum(dim=0)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        fbeta = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.eps)
        return 1 - fbeta[self.fault_class_idx]

    def _fpr_loss(self, probs: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        """Calculates the False Positive Rate of predicting 'Fault' for 'Normal' samples."""
        normal_mask = one_hot[:, self.normal_class_idx] == 1
        if not normal_mask.any():
            return torch.tensor(0.0, device=probs.device)
        
        false_positives = probs[normal_mask, self.fault_class_idx].sum()
        true_negatives = (1 - probs[normal_mask, self.fault_class_idx]).sum()
        return false_positives / (false_positives + true_negatives + self.eps)

    def _orthogonal_loss(self, class_rule_vectors: List[torch.Tensor]) -> torch.Tensor:
        """Encourages rule vectors from different classes to be dissimilar (orthogonal)."""
        loss = 0.0
        num_pairs = 0
        for i in range(len(class_rule_vectors)):
            for j in range(i + 1, len(class_rule_vectors)):
                v1 = F.normalize(class_rule_vectors[i], p=2, dim=0)
                v2 = F.normalize(class_rule_vectors[j], p=2, dim=0)
                loss += torch.abs(torch.dot(v1, v2)) # Cosine similarity
                num_pairs += 1
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=loss.device)


class NeuroSymbolicClassifier(nn.Module):
    """
    The main model, integrating class-specific rule banks with a Transformer fusion layer.

    Architecture:
    1.  **Input Normalization**: A BatchNorm layer stabilizes inputs.
    2.  **Class-Specific Rule Banks**: For each class, a set of `InterpretableRule`
        modules are applied to weighted input features.
    3.  **Transformer Fusion**: The outputs of the rule banks are projected and fed
        into a Transformer encoder, which captures complex interactions between rules.
    4.  **Classifier Head**: A final MLP combines the Transformer's output to
        produce class logits.
    """
    def __init__(self, input_dim: int, num_classes: int, config: object):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.rules_per_feature = int(config.RULES_PER_FEATURE)
        hidden_dim = int(config.TRANSFORMER_HIDDEN_DIM)

        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Each class gets its own independent set of rules for every feature
        self.class_rule_extractors = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([InterpretableRule() for _ in range(self.rules_per_feature)])
                for _ in range(input_dim)
            ]) for _ in range(num_classes)
        ])
        
        # Learnable weights to determine feature importance for each class
        self.feature_importance = nn.Parameter(torch.ones(num_classes, input_dim))

        # Transformer components
        self.rules_projection = nn.Linear(input_dim * self.rules_per_feature, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=int(config.TRANSFORMER_HEADS),
            dim_feedforward=hidden_dim * 4, dropout=float(config.DROPOUT_RATE),
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.TRANSFORMER_LAYERS))
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_classes),
            nn.Linear(hidden_dim * num_classes, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.DROPOUT_RATE) * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Defines the forward pass of the model."""
        x_norm = self.input_norm(x) if x.size(0) > 1 else x
        
        class_features, rule_vectors = [], []
        importance_weights = F.softmax(self.feature_importance, dim=1)

        # 1. Process inputs through class-specific rule banks
        for c in range(self.num_classes):
            x_weighted = x_norm * importance_weights[c].unsqueeze(0)
            rule_outputs = [
                rule(x_weighted[:, i:i+1])
                for i in range(self.input_dim)
                for rule in self.class_rule_extractors[c][i]
            ]
            all_rules_activated = torch.cat(rule_outputs, dim=1) # Shape: [Batch, N_Rules]
            
            # Aggregate rule activations for the orthogonal loss
            rule_vectors.append(all_rules_activated.mean(dim=0))
            
            # Project activated rules into the transformer's dimension
            class_features.append(self.rules_projection(all_rules_activated)) # Shape: [Batch, Hidden_Dim]

        # 2. Fuse rule features with the Transformer
        transformer_input = torch.stack(class_features, dim=1) # Shape: [Batch, N_Classes, Hidden_Dim]
        transformer_output = self.transformer_encoder(transformer_input)
        
        # 3. Classify using the fused representation
        fused_features = transformer_output.view(x.size(0), -1) # Flatten
        logits = self.classifier(fused_features)
        
        # Apply temperature scaling for calibration
        temp = torch.clamp(self.temperature, min=0.5, max=2.0)
        return logits / temp, rule_vectors

    @torch.no_grad()
    def extract_rules(self, scaler: RobustScaler, feature_names: List[str], class_names_map: Dict[int, str]) -> Dict[str, Any]:
        """Extracts the learned rules in a human-readable format."""
        self.eval()
        importance_scores = F.softmax(self.feature_importance, dim=1).cpu().numpy()
        output: Dict[str, Any] = {}
        
        for class_idx, class_name in class_names_map.items():
            # Find top K most important features for this class
            top_feature_indices = np.argsort(importance_scores[class_idx])[::-1][:5]
            
            rules_for_class: Dict[str, List[str]] = {}
            for feat_idx in top_feature_indices:
                feature_name = feature_names[feat_idx]
                conditions = [
                    rule.describe(feature_name, scaler, feat_idx)
                    for rule in self.class_rule_extractors[class_idx][feat_idx]
                ]
                rules_for_class.setdefault(feature_name, []).extend(conditions)
            output[class_name] = rules_for_class
            
        return output
