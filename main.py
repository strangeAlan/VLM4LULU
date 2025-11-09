import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from model import CustomDataset, MLP,MLP2,SimpleMLP,MultiTaskModel,FocalLoss,MultiTaskAttn
from load_data import load_dataset,split_dataset
import argparse
from sklearn.metrics import f1_score
import  numpy as np
import random
from tqdm import tqdm
def set_random_seed(seed=42):
    """固定所有可能影响结果的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def main(args):
    set_random_seed(args.seed)
    # device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载数据
    features, label1, label2, label3 = load_dataset(args.data_path)
    
    X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = split_dataset(
    features, label1, label2, label3, test_size=0.2, random_state=42
)
    train_dataset = CustomDataset(X_train, y1_train, y2_train, y3_train)
    test_dataset = CustomDataset(X_test, y1_test, y2_test, y3_test)
    # test_dataset = CustomDataset(features, label1, label2, label3)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

   
    input_size = features.shape[1]
    # output_size = 2  
    model_dict = {
        "mlp": MLP,
        "mlp2": MLP2,
        "SimpleMLP":SimpleMLP,
        "MTM":MultiTaskModel,
        "att":MultiTaskAttn
    }
    # model = MLP(input_size, args.hidden_size, output_size).to(device)
    model = model_dict[args.model](
                input_size=input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                dropout_rate=args.dropout  
            ).to(device)
    criterion = nn.CrossEntropyLoss()  
    # criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)
    lambda_l1 = 1e-4  # 正则化系数，可以调节
    # 训练阶段
    epoch_loop = tqdm(
    range(args.epochs), 
    desc="Total Training", 
    unit="epoch"
    )
    for epoch in epoch_loop:
        model.train()
        train_total_loss = 0.0
        # train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for inputs, l1, l2, l3 in train_loader:
            inputs = inputs.to(device).float()
            l1, l2, l3 = l1.to(device).long(), l2.to(device).long(), l3.to(device).long()
            # l1, l2, l3 = l1.to(device).float(), l2.to(device).float(), l3.to(device).float()

            if args.model !="att":
                outputs1, outputs2, outputs3 = model(inputs)  

                # loss1 = criterion(outputs1.view(-1), l1)
                # loss2 = criterion(outputs2.view(-1), l1)
                # loss3 = criterion(outputs3.view(-1), l1)
                loss1 = criterion(outputs1, l1)
                loss2 = criterion(outputs2, l2)
                loss3 = criterion(outputs3, l3)
            else:
                # print("input.shape:",inputs.shape)
                # continue
                outs, attn_weights = model(inputs)

                loss1 = criterion(outs[0], l1)*attn_weights[:, 0].mean()
                loss2 = criterion(outs[1], l2)*attn_weights[:, 1].mean()
                loss3 = criterion(outs[2], l3)*attn_weights[:, 2].mean()


            loss = loss1 + loss2 + loss3  # 总损失
            if args.use_l1:
                l1_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_reg += torch.sum(torch.abs(param))
                loss += l1_reg#添加L1正则化
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item() * inputs.size(0)  # 按批次大小累计
        # 计算训练集平均损失
        # train_avg_loss = train_total_loss / len(train_dataset)
        # print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_avg_loss:.4f}")
        

    # 测试阶段（计算acc和f1-score）
    model.eval()
    all_pred1, all_pred2, all_pred3 = [], [], []
    all_true1, all_true2, all_true3 = [], [], []
    
    with torch.no_grad():
        for inputs, l1, l2, l3 in test_loader:
            inputs = inputs.to(device).float()
            l1, l2, l3 = l1.to(device).long(), l2.to(device).long(), l3.to(device).long()
            if args.model !="att":
                outputs1, outputs2, outputs3 = model(inputs)
                pred1 = torch.argmax(outputs1, dim=1)
                pred2 = torch.argmax(outputs2, dim=1)
                pred3 = torch.argmax(outputs3, dim=1)
            else:
                outs, combined = model(inputs)
                pred1 = torch.argmax(outs[0], dim=1)
                pred2 = torch.argmax(outs[1], dim=1)
                pred3 = torch.argmax(outs[2], dim=1)

            

            # 收集所有预测和真实标签（转为numpy用于计算f1）
            all_pred1.extend(pred1.cpu().numpy())
            all_pred2.extend(pred2.cpu().numpy())
            all_pred3.extend(pred3.cpu().numpy())
            all_true1.extend(l1.cpu().numpy())
            all_true2.extend(l2.cpu().numpy())
            all_true3.extend(l3.cpu().numpy())

    # 计算各标签的acc和f1-score
    acc1 = np.mean(np.array(all_pred1) == np.array(all_true1))
    acc2 = np.mean(np.array(all_pred2) == np.array(all_true2))
    acc3 = np.mean(np.array(all_pred3) == np.array(all_true3))
    avg_acc = (acc1 + acc2 + acc3) / 3

    f1_1 = f1_score(all_true1, all_pred1, average='binary')  # 二分类用binary
    f1_2 = f1_score(all_true2, all_pred2, average='binary')
    f1_3 = f1_score(all_true3, all_pred3, average='binary')
    avg_f1 = (f1_1 + f1_2 + f1_3) / 3

    # 输出结果
    print("\nTest Results:")
    print(f"Label1 - Acc: {acc1:.2%}, F1: {f1_1:.4f}")
    print(f"Label2 - Acc: {acc2:.2%}, F1: {f1_2:.4f}")
    print(f"Label3 - Acc: {acc3:.2%}, F1: {f1_3:.4f}")
    print(f"平均Accuracy: {avg_acc:.2%}, 平均F1-score: {avg_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-label Binary Classification with MLP (3 labels, each 2 classes)")
    # 数据参数
    parser.add_argument("--data_path", type=str, default="data.pt", help="Path to dataset")
    parser.add_argument("--gpu_num", type=int, default="3", help="gpu num")
    parser.add_argument("--seed", type=int, default="1", help="seed")
    # 训练参数
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--hidden_size", type=int, default=16, help="Number of hidden units")
    parser.add_argument("--output_size", type=int, default=2, help="out size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "mlp2","SimpleMLP","MTM","att"], 
                      help="选择模型）")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
    parser.add_argument("--use_l1", type=bool, default=False, help="L1正则化")
    
    args = parser.parse_args()
    print(args)
    main(args)