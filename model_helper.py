import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd


def train_model(model, device, dataloader, lr, weight_decay, model_name, epoch_number=20, save_flg=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_loss_list = []

    for epoch in range(epoch_number):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            inputs, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        epoch_loss_list.append([epoch + 1, avg_loss])

    loss_df = pd.DataFrame(epoch_loss_list, columns=['Epoch', 'Loss'])

    if save_flg:
        saved_path = "./models/"
        os.makedirs(saved_path, exist_ok=True)
        PATH = saved_path + f'{model_name}_{lr}_{weight_decay}.pth'
        torch.save(model.state_dict(), PATH)

        loss_path = './loss_results/'
        model_path = f'{model_name}_{lr}_{weight_decay}.csv'

        os.makedirs(loss_path, exist_ok=True)

        if os.path.exists(loss_path + model_path):
            loss_df.to_csv(loss_path + model_path, mode='a', header=False, index=False)
        else:
            loss_df.to_csv(loss_path + model_path, mode='w', header=True, index=False)

    return model


def predict_model(model, device, classes, test_loader, lr, weight_decay, model_name, tested_parameter,save_flg = True):
    columns = [
        "model_name",
        "lr",
        "weight_decay",
        "acc_total",
        "acc_yes",
        "acc_no",
        "acc_up",
        "acc_down",
        "acc_left",
        "acc_right",
        "acc_on",
        "acc_off",
        "acc_stop",
        "acc_go",
        "acc_silence",
        "acc_unknown"
    ]

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(12)]
        n_class_samples = [0 for _ in range(12)]

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

    acc_total = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network {model_name} with lr = {lr} and weight decay = {weight_decay}: {acc_total:.2f}%')
    acc_per_class = []
    for i in range(12):
        class_acc = round(100.0 * n_class_correct[i] / n_class_samples[i], 2)
        acc_per_class.append(class_acc)
        print(f'Accuracy of class "{classes[i]}": {class_acc:.2f}%')

    new_row = pd.DataFrame([[
        model_name, lr, weight_decay, acc_total, *acc_per_class
    ]], columns=columns)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()

    if save_flg:
        hyper_path = './hyper_results/'
        model_path = f'{model_name}_{tested_parameter}'
        os.makedirs(hyper_path, exist_ok=True)
        hyper_full_path = hyper_path + model_path + '.csv'
        if os.path.exists(hyper_full_path):
            new_row.to_csv(hyper_full_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(hyper_full_path, mode='w', header=True, index=False)

        plot_path = "./matrix_plots/"
        model_path_fig = f'{model_name}_{lr}_{weight_decay}'
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(plot_path + model_path_fig + '.png')
