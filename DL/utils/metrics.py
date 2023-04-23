import torchmetrics


class MetricCalculator:
    def __init__(self, metrics=["acc", "f1", "recall"], average="macro", num_classes=7):
        self.acc = torchmetrics.Accuracy(average=average, num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(average=average, num_classes=num_classes)
        self.recall = torchmetrics.Recall(average=average, num_classes=num_classes)
        
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        for m in metrics:
            assert m in self.__dict__.keys(), "acc, f1, recall are you only choices"
        
        self.metrics = metrics
        
    def __call__(self, preds, targets):
        results = {}
        for k in self.metrics:
            self.__dict__[k](preds, targets)
            results[k] = self.__dict__[k].compute()
            
        return results