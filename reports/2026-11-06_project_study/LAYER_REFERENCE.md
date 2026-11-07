# HybridBathNet layer-by-layer reference

Generated from strict checkpoint reload and one evaluation forward pass. Input example batch size is1; model weights are not modified. Registered layers can exist without executing in a particular configuration.

## Full v1

| Module | Type/specification | Input | Output | Parameters | Executed |
|---|---|---|---|---|---|
| encoder.stem1.0 | Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 16, 64, 64]] | [1, 32, 64, 64] | 4608 | True |
| encoder.stem1.1 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.stem1.2 | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.stem1.3 | Dropout2d(p=0.1, inplace=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.stem1.4 | Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 9216 | True |
| encoder.stem1.5 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.stem1.6 | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.stem1.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 32, 64, 64]] | [1, 32, 1, 1] | 0 | True |
| encoder.stem1.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 32, 1, 1]] | [1, 32] | 0 | True |
| encoder.stem1.7.se.2 | Linear(in_features=32, out_features=4, bias=True) | [[1, 32]] | [1, 4] | 132 | True |
| encoder.stem1.7.se.3 | ReLU(inplace=True) | [[1, 4]] | [1, 4] | 0 | True |
| encoder.stem1.7.se.4 | Linear(in_features=4, out_features=32, bias=True) | [[1, 4]] | [1, 32] | 160 | True |
| encoder.stem1.7.se.5 | Sigmoid() | [[1, 32]] | [1, 32] | 0 | True |
| encoder.res1.conv.0 | Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 9216 | True |
| encoder.res1.conv.1 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.res1.conv.2 | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.res1.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.res1.conv.4 | Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 9216 | True |
| encoder.res1.conv.5 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.res1.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 32, 64, 64]] | [1, 32, 1, 1] | 0 | True |
| encoder.res1.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 32, 1, 1]] | [1, 32] | 0 | True |
| encoder.res1.se.se.2 | Linear(in_features=32, out_features=4, bias=True) | [[1, 32]] | [1, 4] | 132 | True |
| encoder.res1.se.se.3 | ReLU(inplace=True) | [[1, 4]] | [1, 4] | 0 | True |
| encoder.res1.se.se.4 | Linear(in_features=4, out_features=32, bias=True) | [[1, 4]] | [1, 32] | 160 | True |
| encoder.res1.se.se.5 | Sigmoid() | [[1, 32]] | [1, 32] | 0 | True |
| encoder.res1.act | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.film1.net | Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 64, 64]] | [1, 64, 64, 64] | 256 | True |
| encoder.pool1 | MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) | [[1, 32, 64, 64]] | [1, 32, 32, 32] | 0 | True |
| encoder.stem2.0 | Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 32, 32]] | [1, 64, 32, 32] | 18432 | True |
| encoder.stem2.1 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.stem2.2 | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.stem2.3 | Dropout2d(p=0.1, inplace=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.stem2.4 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 36864 | True |
| encoder.stem2.5 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.stem2.6 | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.stem2.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 64, 32, 32]] | [1, 64, 1, 1] | 0 | True |
| encoder.stem2.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 64, 1, 1]] | [1, 64] | 0 | True |
| encoder.stem2.7.se.2 | Linear(in_features=64, out_features=8, bias=True) | [[1, 64]] | [1, 8] | 520 | True |
| encoder.stem2.7.se.3 | ReLU(inplace=True) | [[1, 8]] | [1, 8] | 0 | True |
| encoder.stem2.7.se.4 | Linear(in_features=8, out_features=64, bias=True) | [[1, 8]] | [1, 64] | 576 | True |
| encoder.stem2.7.se.5 | Sigmoid() | [[1, 64]] | [1, 64] | 0 | True |
| encoder.res2.conv.0 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 36864 | True |
| encoder.res2.conv.1 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.res2.conv.2 | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.res2.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.res2.conv.4 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 36864 | True |
| encoder.res2.conv.5 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.res2.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 64, 32, 32]] | [1, 64, 1, 1] | 0 | True |
| encoder.res2.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 64, 1, 1]] | [1, 64] | 0 | True |
| encoder.res2.se.se.2 | Linear(in_features=64, out_features=8, bias=True) | [[1, 64]] | [1, 8] | 520 | True |
| encoder.res2.se.se.3 | ReLU(inplace=True) | [[1, 8]] | [1, 8] | 0 | True |
| encoder.res2.se.se.4 | Linear(in_features=8, out_features=64, bias=True) | [[1, 8]] | [1, 64] | 576 | True |
| encoder.res2.se.se.5 | Sigmoid() | [[1, 64]] | [1, 64] | 0 | True |
| encoder.res2.act | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.film2.net | Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 32, 32]] | [1, 128, 32, 32] | 512 | True |
| encoder.pool2 | MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) | [[1, 64, 32, 32]] | [1, 64, 16, 16] | 0 | True |
| encoder.stem3.0 | Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 16, 16]] | [1, 96, 16, 16] | 55296 | True |
| encoder.stem3.1 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.stem3.2 | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.stem3.3 | Dropout2d(p=0.1, inplace=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.stem3.4 | Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 82944 | True |
| encoder.stem3.5 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.stem3.6 | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.stem3.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 96, 16, 16]] | [1, 96, 1, 1] | 0 | True |
| encoder.stem3.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 96, 1, 1]] | [1, 96] | 0 | True |
| encoder.stem3.7.se.2 | Linear(in_features=96, out_features=12, bias=True) | [[1, 96]] | [1, 12] | 1164 | True |
| encoder.stem3.7.se.3 | ReLU(inplace=True) | [[1, 12]] | [1, 12] | 0 | True |
| encoder.stem3.7.se.4 | Linear(in_features=12, out_features=96, bias=True) | [[1, 12]] | [1, 96] | 1248 | True |
| encoder.stem3.7.se.5 | Sigmoid() | [[1, 96]] | [1, 96] | 0 | True |
| encoder.res3.conv.0 | Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 82944 | True |
| encoder.res3.conv.1 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.res3.conv.2 | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.res3.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.res3.conv.4 | Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 82944 | True |
| encoder.res3.conv.5 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.res3.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 96, 16, 16]] | [1, 96, 1, 1] | 0 | True |
| encoder.res3.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 96, 1, 1]] | [1, 96] | 0 | True |
| encoder.res3.se.se.2 | Linear(in_features=96, out_features=12, bias=True) | [[1, 96]] | [1, 12] | 1164 | True |
| encoder.res3.se.se.3 | ReLU(inplace=True) | [[1, 12]] | [1, 12] | 0 | True |
| encoder.res3.se.se.4 | Linear(in_features=12, out_features=96, bias=True) | [[1, 12]] | [1, 96] | 1248 | True |
| encoder.res3.se.se.5 | Sigmoid() | [[1, 96]] | [1, 96] | 0 | True |
| encoder.res3.act | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.film3.net | Conv2d(3, 192, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 16, 16]] | [1, 192, 16, 16] | 768 | True |
| encoder.pool3 | MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) | [[1, 96, 16, 16]] | [1, 96, 8, 8] | 0 | True |
| encoder.bottleneck.0 | Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 8, 8]] | [1, 128, 8, 8] | 110592 | True |
| encoder.bottleneck.1 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.bottleneck.2 | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.bottleneck.3 | Dropout2d(p=0.1, inplace=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.bottleneck.4 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 147456 | True |
| encoder.bottleneck.5 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.bottleneck.6 | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.bottleneck.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 128, 8, 8]] | [1, 128, 1, 1] | 0 | True |
| encoder.bottleneck.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 128, 1, 1]] | [1, 128] | 0 | True |
| encoder.bottleneck.7.se.2 | Linear(in_features=128, out_features=16, bias=True) | [[1, 128]] | [1, 16] | 2064 | True |
| encoder.bottleneck.7.se.3 | ReLU(inplace=True) | [[1, 16]] | [1, 16] | 0 | True |
| encoder.bottleneck.7.se.4 | Linear(in_features=16, out_features=128, bias=True) | [[1, 16]] | [1, 128] | 2176 | True |
| encoder.bottleneck.7.se.5 | Sigmoid() | [[1, 128]] | [1, 128] | 0 | True |
| encoder.res_btn.conv.0 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 147456 | True |
| encoder.res_btn.conv.1 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.res_btn.conv.2 | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.res_btn.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.res_btn.conv.4 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 147456 | True |
| encoder.res_btn.conv.5 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.res_btn.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 128, 8, 8]] | [1, 128, 1, 1] | 0 | True |
| encoder.res_btn.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 128, 1, 1]] | [1, 128] | 0 | True |
| encoder.res_btn.se.se.2 | Linear(in_features=128, out_features=16, bias=True) | [[1, 128]] | [1, 16] | 2064 | True |
| encoder.res_btn.se.se.3 | ReLU(inplace=True) | [[1, 16]] | [1, 16] | 0 | True |
| encoder.res_btn.se.se.4 | Linear(in_features=16, out_features=128, bias=True) | [[1, 16]] | [1, 128] | 2176 | True |
| encoder.res_btn.se.se.5 | Sigmoid() | [[1, 128]] | [1, 128] | 0 | True |
| encoder.res_btn.act | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.film_btn.net | Conv2d(3, 256, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 8, 8]] | [1, 256, 8, 8] | 1024 | True |
| decoder.up3 | ConvTranspose2d(128, 96, kernel_size=(2, 2), stride=(2, 2)) | [[1, 128, 8, 8]] | [1, 96, 16, 16] | 49248 | True |
| decoder.dec3.conv.0 | Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 331776 | True |
| decoder.dec3.conv.1 | BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 384 | True |
| decoder.dec3.conv.2 | ReLU(inplace=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 0 | True |
| decoder.dec3.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 0 | True |
| decoder.dec3.conv.4 | Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 331776 | True |
| decoder.dec3.conv.5 | BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 384 | True |
| decoder.dec3.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 192, 16, 16]] | [1, 192, 1, 1] | 0 | True |
| decoder.dec3.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 192, 1, 1]] | [1, 192] | 0 | True |
| decoder.dec3.se.se.2 | Linear(in_features=192, out_features=24, bias=True) | [[1, 192]] | [1, 24] | 4632 | True |
| decoder.dec3.se.se.3 | ReLU(inplace=True) | [[1, 24]] | [1, 24] | 0 | True |
| decoder.dec3.se.se.4 | Linear(in_features=24, out_features=192, bias=True) | [[1, 24]] | [1, 192] | 4800 | True |
| decoder.dec3.se.se.5 | Sigmoid() | [[1, 192]] | [1, 192] | 0 | True |
| decoder.dec3.act | ReLU(inplace=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 0 | True |
| decoder.up2 | ConvTranspose2d(192, 64, kernel_size=(2, 2), stride=(2, 2)) | [[1, 192, 16, 16]] | [1, 64, 32, 32] | 49216 | True |
| decoder.dec2.conv.0 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 147456 | True |
| decoder.dec2.conv.1 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 256 | True |
| decoder.dec2.conv.2 | ReLU(inplace=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 0 | True |
| decoder.dec2.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 0 | True |
| decoder.dec2.conv.4 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 147456 | True |
| decoder.dec2.conv.5 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 256 | True |
| decoder.dec2.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 128, 32, 32]] | [1, 128, 1, 1] | 0 | True |
| decoder.dec2.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 128, 1, 1]] | [1, 128] | 0 | True |
| decoder.dec2.se.se.2 | Linear(in_features=128, out_features=16, bias=True) | [[1, 128]] | [1, 16] | 2064 | True |
| decoder.dec2.se.se.3 | ReLU(inplace=True) | [[1, 16]] | [1, 16] | 0 | True |
| decoder.dec2.se.se.4 | Linear(in_features=16, out_features=128, bias=True) | [[1, 16]] | [1, 128] | 2176 | True |
| decoder.dec2.se.se.5 | Sigmoid() | [[1, 128]] | [1, 128] | 0 | True |
| decoder.dec2.act | ReLU(inplace=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 0 | True |
| decoder.up1 | ConvTranspose2d(128, 32, kernel_size=(2, 2), stride=(2, 2)) | [[1, 128, 32, 32]] | [1, 32, 64, 64] | 16416 | True |
| decoder.dec1.conv.0 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 36864 | True |
| decoder.dec1.conv.1 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 128 | True |
| decoder.dec1.conv.2 | ReLU(inplace=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 0 | True |
| decoder.dec1.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 0 | True |
| decoder.dec1.conv.4 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 36864 | True |
| decoder.dec1.conv.5 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 128 | True |
| decoder.dec1.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 64, 64, 64]] | [1, 64, 1, 1] | 0 | True |
| decoder.dec1.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 64, 1, 1]] | [1, 64] | 0 | True |
| decoder.dec1.se.se.2 | Linear(in_features=64, out_features=8, bias=True) | [[1, 64]] | [1, 8] | 520 | True |
| decoder.dec1.se.se.3 | ReLU(inplace=True) | [[1, 8]] | [1, 8] | 0 | True |
| decoder.dec1.se.se.4 | Linear(in_features=8, out_features=64, bias=True) | [[1, 8]] | [1, 64] | 576 | True |
| decoder.dec1.se.se.5 | Sigmoid() | [[1, 64]] | [1, 64] | 0 | True |
| decoder.dec1.act | ReLU(inplace=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 0 | True |
| decoder.depth_conv | Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)) | [[1, 64, 64, 64]] | [1, 1, 64, 64] | 65 | True |
| decoder.logvar_conv | Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)) | [[1, 64, 64, 64]] | [1, 1, 64, 64] | 65 | True |
| cond_proj | Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 64, 64]] | [1, 3, 64, 64] | 12 | True |
| recon_head.kl_head.0 | Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 32, 64, 64]] | [1, 16, 64, 64] | 4624 | True |
| recon_head.kl_head.1 | ReLU(inplace=True) | [[1, 16, 64, 64]] | [1, 16, 64, 64] | 0 | True |
| recon_head.kl_head.2 | Conv2d(16, 3, kernel_size=(1, 1), stride=(1, 1)) | [[1, 16, 64, 64]] | [1, 3, 64, 64] | 51 | True |
| recon_head.al_head.0 | Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 32, 64, 64]] | [1, 16, 64, 64] | 4624 | True |
| recon_head.al_head.1 | ReLU(inplace=True) | [[1, 16, 64, 64]] | [1, 16, 64, 64] | 0 | True |
| recon_head.al_head.2 | Conv2d(16, 3, kernel_size=(1, 1), stride=(1, 1)) | [[1, 16, 64, 64]] | [1, 3, 64, 64] | 51 | True |
| gap | AdaptiveAvgPool2d(output_size=1) | None | None | 0 | False |

## Spatial uncertainty member

| Module | Type/specification | Input | Output | Parameters | Executed |
|---|---|---|---|---|---|
| encoder.stem1.0 | Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 16, 64, 64]] | [1, 32, 64, 64] | 4608 | True |
| encoder.stem1.1 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.stem1.2 | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.stem1.3 | Dropout2d(p=0.1, inplace=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.stem1.4 | Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 9216 | True |
| encoder.stem1.5 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.stem1.6 | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.stem1.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 32, 64, 64]] | [1, 32, 1, 1] | 0 | True |
| encoder.stem1.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 32, 1, 1]] | [1, 32] | 0 | True |
| encoder.stem1.7.se.2 | Linear(in_features=32, out_features=4, bias=True) | [[1, 32]] | [1, 4] | 132 | True |
| encoder.stem1.7.se.3 | ReLU(inplace=True) | [[1, 4]] | [1, 4] | 0 | True |
| encoder.stem1.7.se.4 | Linear(in_features=4, out_features=32, bias=True) | [[1, 4]] | [1, 32] | 160 | True |
| encoder.stem1.7.se.5 | Sigmoid() | [[1, 32]] | [1, 32] | 0 | True |
| encoder.res1.conv.0 | Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 9216 | True |
| encoder.res1.conv.1 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.res1.conv.2 | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.res1.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.res1.conv.4 | Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 9216 | True |
| encoder.res1.conv.5 | BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| encoder.res1.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 32, 64, 64]] | [1, 32, 1, 1] | 0 | True |
| encoder.res1.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 32, 1, 1]] | [1, 32] | 0 | True |
| encoder.res1.se.se.2 | Linear(in_features=32, out_features=4, bias=True) | [[1, 32]] | [1, 4] | 132 | True |
| encoder.res1.se.se.3 | ReLU(inplace=True) | [[1, 4]] | [1, 4] | 0 | True |
| encoder.res1.se.se.4 | Linear(in_features=4, out_features=32, bias=True) | [[1, 4]] | [1, 32] | 160 | True |
| encoder.res1.se.se.5 | Sigmoid() | [[1, 32]] | [1, 32] | 0 | True |
| encoder.res1.act | ReLU(inplace=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| encoder.film1.net | Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 64, 64]] | [1, 64, 64, 64] | 256 | True |
| encoder.pool1 | MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) | [[1, 32, 64, 64]] | [1, 32, 32, 32] | 0 | True |
| encoder.stem2.0 | Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 32, 32, 32]] | [1, 64, 32, 32] | 18432 | True |
| encoder.stem2.1 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.stem2.2 | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.stem2.3 | Dropout2d(p=0.1, inplace=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.stem2.4 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 36864 | True |
| encoder.stem2.5 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.stem2.6 | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.stem2.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 64, 32, 32]] | [1, 64, 1, 1] | 0 | True |
| encoder.stem2.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 64, 1, 1]] | [1, 64] | 0 | True |
| encoder.stem2.7.se.2 | Linear(in_features=64, out_features=8, bias=True) | [[1, 64]] | [1, 8] | 520 | True |
| encoder.stem2.7.se.3 | ReLU(inplace=True) | [[1, 8]] | [1, 8] | 0 | True |
| encoder.stem2.7.se.4 | Linear(in_features=8, out_features=64, bias=True) | [[1, 8]] | [1, 64] | 576 | True |
| encoder.stem2.7.se.5 | Sigmoid() | [[1, 64]] | [1, 64] | 0 | True |
| encoder.res2.conv.0 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 36864 | True |
| encoder.res2.conv.1 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.res2.conv.2 | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.res2.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.res2.conv.4 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 36864 | True |
| encoder.res2.conv.5 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 128 | True |
| encoder.res2.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 64, 32, 32]] | [1, 64, 1, 1] | 0 | True |
| encoder.res2.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 64, 1, 1]] | [1, 64] | 0 | True |
| encoder.res2.se.se.2 | Linear(in_features=64, out_features=8, bias=True) | [[1, 64]] | [1, 8] | 520 | True |
| encoder.res2.se.se.3 | ReLU(inplace=True) | [[1, 8]] | [1, 8] | 0 | True |
| encoder.res2.se.se.4 | Linear(in_features=8, out_features=64, bias=True) | [[1, 8]] | [1, 64] | 576 | True |
| encoder.res2.se.se.5 | Sigmoid() | [[1, 64]] | [1, 64] | 0 | True |
| encoder.res2.act | ReLU(inplace=True) | [[1, 64, 32, 32]] | [1, 64, 32, 32] | 0 | True |
| encoder.film2.net | Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 32, 32]] | [1, 128, 32, 32] | 512 | True |
| encoder.pool2 | MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) | [[1, 64, 32, 32]] | [1, 64, 16, 16] | 0 | True |
| encoder.stem3.0 | Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 16, 16]] | [1, 96, 16, 16] | 55296 | True |
| encoder.stem3.1 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.stem3.2 | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.stem3.3 | Dropout2d(p=0.1, inplace=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.stem3.4 | Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 82944 | True |
| encoder.stem3.5 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.stem3.6 | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.stem3.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 96, 16, 16]] | [1, 96, 1, 1] | 0 | True |
| encoder.stem3.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 96, 1, 1]] | [1, 96] | 0 | True |
| encoder.stem3.7.se.2 | Linear(in_features=96, out_features=12, bias=True) | [[1, 96]] | [1, 12] | 1164 | True |
| encoder.stem3.7.se.3 | ReLU(inplace=True) | [[1, 12]] | [1, 12] | 0 | True |
| encoder.stem3.7.se.4 | Linear(in_features=12, out_features=96, bias=True) | [[1, 12]] | [1, 96] | 1248 | True |
| encoder.stem3.7.se.5 | Sigmoid() | [[1, 96]] | [1, 96] | 0 | True |
| encoder.res3.conv.0 | Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 82944 | True |
| encoder.res3.conv.1 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.res3.conv.2 | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.res3.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.res3.conv.4 | Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 82944 | True |
| encoder.res3.conv.5 | BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 192 | True |
| encoder.res3.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 96, 16, 16]] | [1, 96, 1, 1] | 0 | True |
| encoder.res3.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 96, 1, 1]] | [1, 96] | 0 | True |
| encoder.res3.se.se.2 | Linear(in_features=96, out_features=12, bias=True) | [[1, 96]] | [1, 12] | 1164 | True |
| encoder.res3.se.se.3 | ReLU(inplace=True) | [[1, 12]] | [1, 12] | 0 | True |
| encoder.res3.se.se.4 | Linear(in_features=12, out_features=96, bias=True) | [[1, 12]] | [1, 96] | 1248 | True |
| encoder.res3.se.se.5 | Sigmoid() | [[1, 96]] | [1, 96] | 0 | True |
| encoder.res3.act | ReLU(inplace=True) | [[1, 96, 16, 16]] | [1, 96, 16, 16] | 0 | True |
| encoder.film3.net | Conv2d(3, 192, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 16, 16]] | [1, 192, 16, 16] | 768 | True |
| encoder.pool3 | MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) | [[1, 96, 16, 16]] | [1, 96, 8, 8] | 0 | True |
| encoder.bottleneck.0 | Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 96, 8, 8]] | [1, 128, 8, 8] | 110592 | True |
| encoder.bottleneck.1 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.bottleneck.2 | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.bottleneck.3 | Dropout2d(p=0.1, inplace=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.bottleneck.4 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 147456 | True |
| encoder.bottleneck.5 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.bottleneck.6 | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.bottleneck.7.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 128, 8, 8]] | [1, 128, 1, 1] | 0 | True |
| encoder.bottleneck.7.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 128, 1, 1]] | [1, 128] | 0 | True |
| encoder.bottleneck.7.se.2 | Linear(in_features=128, out_features=16, bias=True) | [[1, 128]] | [1, 16] | 2064 | True |
| encoder.bottleneck.7.se.3 | ReLU(inplace=True) | [[1, 16]] | [1, 16] | 0 | True |
| encoder.bottleneck.7.se.4 | Linear(in_features=16, out_features=128, bias=True) | [[1, 16]] | [1, 128] | 2176 | True |
| encoder.bottleneck.7.se.5 | Sigmoid() | [[1, 128]] | [1, 128] | 0 | True |
| encoder.res_btn.conv.0 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 147456 | True |
| encoder.res_btn.conv.1 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.res_btn.conv.2 | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.res_btn.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.res_btn.conv.4 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 147456 | True |
| encoder.res_btn.conv.5 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 256 | True |
| encoder.res_btn.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 128, 8, 8]] | [1, 128, 1, 1] | 0 | True |
| encoder.res_btn.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 128, 1, 1]] | [1, 128] | 0 | True |
| encoder.res_btn.se.se.2 | Linear(in_features=128, out_features=16, bias=True) | [[1, 128]] | [1, 16] | 2064 | True |
| encoder.res_btn.se.se.3 | ReLU(inplace=True) | [[1, 16]] | [1, 16] | 0 | True |
| encoder.res_btn.se.se.4 | Linear(in_features=16, out_features=128, bias=True) | [[1, 16]] | [1, 128] | 2176 | True |
| encoder.res_btn.se.se.5 | Sigmoid() | [[1, 128]] | [1, 128] | 0 | True |
| encoder.res_btn.act | ReLU(inplace=True) | [[1, 128, 8, 8]] | [1, 128, 8, 8] | 0 | True |
| encoder.film_btn.net | Conv2d(3, 256, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 8, 8]] | [1, 256, 8, 8] | 1024 | True |
| decoder.up3 | ConvTranspose2d(128, 96, kernel_size=(2, 2), stride=(2, 2)) | [[1, 128, 8, 8]] | [1, 96, 16, 16] | 49248 | True |
| decoder.dec3.conv.0 | Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 331776 | True |
| decoder.dec3.conv.1 | BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 384 | True |
| decoder.dec3.conv.2 | ReLU(inplace=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 0 | True |
| decoder.dec3.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 0 | True |
| decoder.dec3.conv.4 | Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 331776 | True |
| decoder.dec3.conv.5 | BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 384 | True |
| decoder.dec3.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 192, 16, 16]] | [1, 192, 1, 1] | 0 | True |
| decoder.dec3.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 192, 1, 1]] | [1, 192] | 0 | True |
| decoder.dec3.se.se.2 | Linear(in_features=192, out_features=24, bias=True) | [[1, 192]] | [1, 24] | 4632 | True |
| decoder.dec3.se.se.3 | ReLU(inplace=True) | [[1, 24]] | [1, 24] | 0 | True |
| decoder.dec3.se.se.4 | Linear(in_features=24, out_features=192, bias=True) | [[1, 24]] | [1, 192] | 4800 | True |
| decoder.dec3.se.se.5 | Sigmoid() | [[1, 192]] | [1, 192] | 0 | True |
| decoder.dec3.act | ReLU(inplace=True) | [[1, 192, 16, 16]] | [1, 192, 16, 16] | 0 | True |
| decoder.up2 | ConvTranspose2d(192, 64, kernel_size=(2, 2), stride=(2, 2)) | [[1, 192, 16, 16]] | [1, 64, 32, 32] | 49216 | True |
| decoder.dec2.conv.0 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 147456 | True |
| decoder.dec2.conv.1 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 256 | True |
| decoder.dec2.conv.2 | ReLU(inplace=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 0 | True |
| decoder.dec2.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 0 | True |
| decoder.dec2.conv.4 | Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 147456 | True |
| decoder.dec2.conv.5 | BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 256 | True |
| decoder.dec2.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 128, 32, 32]] | [1, 128, 1, 1] | 0 | True |
| decoder.dec2.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 128, 1, 1]] | [1, 128] | 0 | True |
| decoder.dec2.se.se.2 | Linear(in_features=128, out_features=16, bias=True) | [[1, 128]] | [1, 16] | 2064 | True |
| decoder.dec2.se.se.3 | ReLU(inplace=True) | [[1, 16]] | [1, 16] | 0 | True |
| decoder.dec2.se.se.4 | Linear(in_features=16, out_features=128, bias=True) | [[1, 16]] | [1, 128] | 2176 | True |
| decoder.dec2.se.se.5 | Sigmoid() | [[1, 128]] | [1, 128] | 0 | True |
| decoder.dec2.act | ReLU(inplace=True) | [[1, 128, 32, 32]] | [1, 128, 32, 32] | 0 | True |
| decoder.up1 | ConvTranspose2d(128, 32, kernel_size=(2, 2), stride=(2, 2)) | [[1, 128, 32, 32]] | [1, 32, 64, 64] | 16416 | True |
| decoder.dec1.conv.0 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 36864 | True |
| decoder.dec1.conv.1 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 128 | True |
| decoder.dec1.conv.2 | ReLU(inplace=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 0 | True |
| decoder.dec1.conv.3 | Dropout2d(p=0.1, inplace=False) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 0 | True |
| decoder.dec1.conv.4 | Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 36864 | True |
| decoder.dec1.conv.5 | BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 128 | True |
| decoder.dec1.se.se.0 | AdaptiveAvgPool2d(output_size=1) | [[1, 64, 64, 64]] | [1, 64, 1, 1] | 0 | True |
| decoder.dec1.se.se.1 | Flatten(start_dim=1, end_dim=-1) | [[1, 64, 1, 1]] | [1, 64] | 0 | True |
| decoder.dec1.se.se.2 | Linear(in_features=64, out_features=8, bias=True) | [[1, 64]] | [1, 8] | 520 | True |
| decoder.dec1.se.se.3 | ReLU(inplace=True) | [[1, 8]] | [1, 8] | 0 | True |
| decoder.dec1.se.se.4 | Linear(in_features=8, out_features=64, bias=True) | [[1, 8]] | [1, 64] | 576 | True |
| decoder.dec1.se.se.5 | Sigmoid() | [[1, 64]] | [1, 64] | 0 | True |
| decoder.dec1.act | ReLU(inplace=True) | [[1, 64, 64, 64]] | [1, 64, 64, 64] | 0 | True |
| decoder.depth_conv | Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)) | [[1, 64, 64, 64]] | [1, 1, 64, 64] | 65 | True |
| decoder.logvar_conv | Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)) | None | None | 65 | False |
| decoder.uncertainty_head.0 | Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 64, 64, 64]] | [1, 32, 64, 64] | 18464 | True |
| decoder.uncertainty_head.1 | GroupNorm(8, 32, eps=1e-05, affine=True) | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 64 | True |
| decoder.uncertainty_head.2 | SiLU() | [[1, 32, 64, 64]] | [1, 32, 64, 64] | 0 | True |
| decoder.uncertainty_head.3 | Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 32, 64, 64]] | [1, 16, 64, 64] | 4624 | True |
| decoder.uncertainty_head.4 | SiLU() | [[1, 16, 64, 64]] | [1, 16, 64, 64] | 0 | True |
| decoder.uncertainty_head.5 | Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1)) | [[1, 16, 64, 64]] | [1, 1, 64, 64] | 17 | True |
| cond_proj | Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)) | [[1, 3, 64, 64]] | [1, 3, 64, 64] | 12 | True |
| recon_head.kl_head.0 | Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 32, 64, 64]] | [1, 16, 64, 64] | 4624 | True |
| recon_head.kl_head.1 | ReLU(inplace=True) | [[1, 16, 64, 64]] | [1, 16, 64, 64] | 0 | True |
| recon_head.kl_head.2 | Conv2d(16, 3, kernel_size=(1, 1), stride=(1, 1)) | [[1, 16, 64, 64]] | [1, 3, 64, 64] | 51 | True |
| recon_head.al_head.0 | Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 32, 64, 64]] | [1, 16, 64, 64] | 4624 | True |
| recon_head.al_head.1 | ReLU(inplace=True) | [[1, 16, 64, 64]] | [1, 16, 64, 64] | 0 | True |
| recon_head.al_head.2 | Conv2d(16, 3, kernel_size=(1, 1), stride=(1, 1)) | [[1, 16, 64, 64]] | [1, 3, 64, 64] | 51 | True |
| gap | AdaptiveAvgPool2d(output_size=1) | None | None | 0 | False |

## OOF correction head

| Module | Type/specification | Input | Output | Parameters | Executed |
|---|---|---|---|---|---|
| image.0 | Conv2d(9, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 9, 64, 64]] | [1, 16, 64, 64] | 1312 | True |
| image.1 | GroupNorm(4, 16, eps=1e-05, affine=True) | [[1, 16, 64, 64]] | [1, 16, 64, 64] | 32 | True |
| image.2 | SiLU() | [[1, 16, 64, 64]] | [1, 16, 64, 64] | 0 | True |
| image.3 | AvgPool2d(kernel_size=2, stride=2, padding=0) | [[1, 16, 64, 64]] | [1, 16, 32, 32] | 0 | True |
| image.4 | Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) | [[1, 16, 32, 32]] | [1, 16, 32, 32] | 2320 | True |
| image.5 | GroupNorm(4, 16, eps=1e-05, affine=True) | [[1, 16, 32, 32]] | [1, 16, 32, 32] | 32 | True |
| image.6 | SiLU() | [[1, 16, 32, 32]] | [1, 16, 32, 32] | 0 | True |
| image.7 | AdaptiveAvgPool2d(output_size=1) | [[1, 16, 32, 32]] | [1, 16, 1, 1] | 0 | True |
| head.0 | Linear(in_features=29, out_features=24, bias=True) | [[1, 29]] | [1, 24] | 720 | True |
| head.1 | SiLU() | [[1, 24]] | [1, 24] | 0 | True |
| head.2 | Linear(in_features=24, out_features=1, bias=True) | [[1, 24]] | [1, 1] | 25 | True |
