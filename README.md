# Paper
Towards Efficient and Scalable Sharpness-Aware Minimization-CVPR2022: https://arxiv.org/abs/2203.02714

# Usage

```python
Usage
from looksam_wise_layer import LookSAM


model = YourModel()
criterion = YourCriterion()
base_optimizer = YourBaseOptimizer
loader = YourLoader()

optimizer = LookSAM(
    k=10,
    alpha=0.7,
    model=model,
    base_optimizer=base_optimizer,
    rho=0.1,
    **kwargs
)

...

model.train()

for train_index, (samples, targets) in enumerate(loader):
    ...

    loss = criterion(model(samples), targets)
    loss.backward()
    optimizer.step(
        t=train_index, 
        samples=samples, 
        targets=targets, 
        zero_sam_grad=True, 
        zero_grad=True
    )
    ...
```


