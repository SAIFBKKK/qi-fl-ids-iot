from src.data.dataloader import create_dataloaders_for_node
from src.common.paths import DATA_DIR


def main():

    node_id = "node1"

    node_dir = DATA_DIR / "processed" / node_id

    train_loader, eval_loader = create_dataloaders_for_node(node_dir)

    print("Dataset loaded")
    print("Train batches:", len(train_loader))
    print("Eval batches:", len(eval_loader))

    for X, y in train_loader:
        print("Batch shape:", X.shape)
        print("Labels shape:", y.shape)
        break


if __name__ == "__main__":
    main()
