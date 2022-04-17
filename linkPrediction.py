def df_metrics_per_seed(models, trains_flow, tests_flow):

  train_loss, train_acc = [], []
  test_loss, test_acc = [], []

  for i in range(len(seeds)):
    train_metrics = models[i].evaluate(trains_flow[i])
    test_metrics = models[i].evaluate(tests_flow[i])

    train_loss.append(train_metrics[0])
    train_acc.append(train_metrics[1])

    test_loss.append(test_metrics[0])
    test_acc.append(test_metrics[1])

  return train_loss, train_acc, test_loss, test_acc
