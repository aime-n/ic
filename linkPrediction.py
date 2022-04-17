def df_metrics_per_seed(models, trains_flow, tests_flow, seeds):

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

def df_seeds(seeds, train_loss, train_acc, test_loss, test_acc):

    df = pd.DataFrame(
      {
      'seeds': seeds,
      'train_loss': train_loss,
      'train_acc': train_acc,
      'test_loss': test_loss,
      'test_acc': test_acc
      }
    )

    df.set_index('seeds', inplace=True)

    return df

def plot_metrics_all_seeds(df, seeds):
    plt.subplots(figsize=(8,6))

    plt.plot(df)
    plt.legend(['train_loss', 'train_acc', 'test_loss', 'test_acc'])
    plt.xlabel('Seeds')
    plt.title('Metrics')
    plt.xticks(seeds)
    plt.show()

    return

def gcn_seeds_metrics_and_plots(G, seeds, test_keep_connected=True):
    hist, models, trains_flow, tests_flow = gcn_seeds(G, seeds, test_keep_connected=test_keep_connected)

    plot_history_all_seeds(hist)

    train_loss, train_acc, test_loss, test_acc = df_metrics_per_seed(models, trains_flow, tests_flow, seeds)

    df = df_seeds(seeds, train_loss, train_acc, test_loss, test_acc)
    print(df)
    print(df.mean())

    plot_metrics_all_seeds(df_gcc, seeds)

def plot_history_all_seeds(hist):
    fig, axs = plt.subplots(2,2, figsize=(12,10))

    for i in hist:
        axs[0,0].plot(i.history['binary_accuracy'])
        axs[0,0].set_title('binary_accuracy')

        axs[0,1].plot(i.history['loss'])
        axs[0,1].set_title('loss')

        axs[1,0].plot(i.history['val_binary_accuracy'])
        axs[1,0].set_title('val_binary_accuracy')

        axs[1,1].plot(i.history['val_loss'])
        axs[1,1].set_title('val_loss')

    plt.show()


def gcn_seeds(G, seeds, test_keep_connected=True):
    '''
    Parameters:
    ----------
    G : StellarGraph
    seeds : list
    '''

    hist = []
    models = []
    trains_flow = []
    tests_flow = []

    for seed in tqdm(seeds):
        print('-'*10, "Seed: ", seed, '-'*10)
        edge_splitter_test = EdgeSplitter(G)

        G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
            p=0.1, method="global", keep_connected=test_keep_connected, seed=seed)

        edge_splitter_train = EdgeSplitter(G_test)

        G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
            p=0.1, method="global", keep_connected=False, seed=seed)

        epochs = 50

        train_gen = FullBatchLinkGenerator(G_train, method="gcn")
        train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

        test_gen = FullBatchLinkGenerator(G_test, method="gcn")
        test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

        gcn = GCN(
            layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
        )

        x_inp, x_out = gcn.in_out_tensors()

        prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

        prediction = keras.layers.Reshape((-1,))(prediction)

        model = keras.Model(inputs=x_inp, outputs=prediction)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=keras.losses.binary_crossentropy,
            metrics=["binary_accuracy"],
        )

        history = model.fit(
            train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
        )

        hist.append(history)
        models.append(model)
        trains_flow.append(train_flow)
        tests_flow.append(test_flow)

    return hist, models, trains_flow, tests_flow


def edges_to_df(KG, rel='relation'):
    df_edges_linguistic = pd.DataFrame()

    source, target, relation = [], [], []
    for edge in KG.edges(data=True):
    source.append(edge[0])
    target.append(edge[1])
    relation.append(edge[2][rel])

    df_edges_linguistic['source'] = source
    df_edges_linguistic['relation'] = relation
    df_edges_linguistic['target'] = target

    return df_edges_linguistic


def matrix_adj(KG):
    adj_matrix = nx.adjacency_matrix(KG).todense()

    df_adj_matrix = pd.DataFrame(adj_matrix)

    df_adj_matrix['index'] = list(KG.nodes)
    df_adj_matrix.set_index('index', inplace=True)
    df_adj_matrix.columns = list(KG.nodes)

    return df_adj_matrix


def create_stellargraph(KG):
    df = edges_to_df(KG)
    matrix = matrix_adj(KG)

    G = sg.StellarGraph(matrix, df[['source', 'target']])

    return G


def gcn_function(SG, seed=42, epochs=50, show_metrics_untrained=False, show_metrics_trained=True, show_history=True):
    edge_splitter_test = EdgeSplitter(SG)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
      p=0.1, method="global", keep_connected=False, seed=seed)

    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
      p=0.1, method="global", keep_connected=False, seed=seed)

    train_gen = FullBatchLinkGenerator(G_train, method="gcn")
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

    test_gen = FullBatchLinkGenerator(G_test, method="gcn")
    test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

    gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3)

    x_inp, x_out = gcn.in_out_tensors()

    prediction = LinkEmbedding(activation="relu", method="ip")(x_out)
    prediction = keras.layers.Reshape((-1,))(prediction)

    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
      optimizer=keras.optimizers.Adam(lr=0.01),
      loss=keras.losses.binary_crossentropy,
      metrics=["binary_accuracy"],)

    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)

    if show_metrics_untrained == True:
        print("\nTrain Set Metrics of the initial (untrained) model:")
        for name, val in zip(model.metrics_names, init_train_metrics):
            print("\t{}: {:0.4f}".format(name, val))

        print("\nTest Set Metrics of the initial (untrained) model:")
        for name, val in zip(model.metrics_names, init_test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

    history = model.fit(
        train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
    )

    if show_history == True:
    print(sg.utils.plot_history(history))

    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)

    if show_metrics_trained == True:
        print("\nTrain Set Metrics of the trained model:")
        for name, val in zip(model.metrics_names, train_metrics):
            print("\t{}: {:0.4f}".format(name, val))
            train_acc = val

        print("\nTest Set Metrics of the trained model:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))
            test_acc = val

    return round(train_acc, 5), round(test_acc, 5)
