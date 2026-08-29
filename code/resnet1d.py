"""
1-D ResNet-34 used for the six bundle branch block detectors in this study.

NOTE ON THE TWO-UNIT OUTPUT HEAD
--------------------------------
The six checkpoints evaluated in this paper carry a two-unit output layer
(`nn.Linear(512, 2)`) and are trained with cross-entropy; the positive-class
probability is `softmax(logits)[:, 1]`. An earlier internal-only audit by the
same author (`this14/bbb-racial-fairness-audit`) ships a single-logit variant
of the same backbone; that file will NOT load the checkpoints used here.
The definition below is the one `run_external_validation.py` uses, and it loads
all six checkpoints with `strict=True`.

Input shape is (B, T, C) — batch, timesteps, leads — and `forward` permutes to
(B, C, T) internally. `in_channels` is 12, 3, or 1 depending on lead set.
"""

import torch.nn as nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch))

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idt = self.downsample(x)
        out += idt
        return self.relu(out)


class ResNet1D34(nn.Module):
    def __init__(self, in_channels=12, num_classes=2, base_width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, 15, 2, 7, bias=False),
            nn.BatchNorm1d(base_width), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1))
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        blocks = [3, 4, 6, 3]
        layers, in_ch = [], base_width
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock1D(in_ch, w, stride=2 if (j == 0 and i > 0) else 1))
                in_ch = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):                      # x: (B, T, C)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# Lead configurations, by channel selection from the stored 12-lead array
# (order: I, II, III, aVR, aVL, aVF, V1..V6)
LEAD_SETS = {
    "1lead":  [1],              # lead II
    "3lead":  [0, 1, 10],       # I, II, V5
    "12lead": list(range(12)),
}

# Group-only inverse-frequency loss weights, w_g = N_total / (K * N_g), K = 5.
# These are the values used to train the three reweighted models.
RACE_WEIGHTS = {
    "Asian":    4.6888,
    "Black":    5.0977,
    "Hispanic": 3.0027,
    "White":    0.2499,
    "Other":    3.8969,
}
