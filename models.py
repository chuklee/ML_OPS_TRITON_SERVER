import torch.nn as nn 
import torch
class cls_model(nn.Module):
    def __init__(self, userCount, itemCount, user_embSize=32, item_embSize=32):
        super().__init__()

        self.m_userEmb = nn.Embedding(userCount, user_embSize)
        self.m_itemEmb = nn.Embedding(itemCount, item_embSize)

        # Updated input size based on actual concatenated item features
        item_input_size = 404 + item_embSize  # 404 from item features + 32 from item embedding = 436

        self.m_modelUser = nn.Sequential(
            nn.Linear(36, 64),  # 4 user features + 32 embedding
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, user_embSize)
        )

        self.m_modelItem = nn.Sequential(
            nn.Linear(item_input_size, 128),  # Updated from 437 to 436
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, item_embSize)
        )

        self.m_modelClassify = nn.Sequential(
            nn.Linear(user_embSize + item_embSize, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, dataUser, dataItem):
        embu = self.m_userEmb(dataUser[:, 0].long())              # [batch_size, 32]
        embi = self.m_itemEmb(dataItem[:, 0].long())              # [batch_size, 32]

        inputU = torch.cat((dataUser[:, 1:].float(), embu), dim=1)  # [batch_size, 36]
        inputI = torch.cat((dataItem[:, 1:].float(), embi), dim=1)  # [batch_size, 436]

        logitsU = self.m_modelUser(inputU)                         # [batch_size, 32]
        logitsI = self.m_modelItem(inputI)                         # [batch_size, 32]

        logits = self.m_modelClassify(torch.cat((logitsU, logitsI), dim=1))  # [batch_size, 2]
        return logits

    def predict(self, userID: torch.tensor):
        """
        Get the similarity between userIDs and all available items
        """
        embu = self.m_userEmb(userID.long())                      # [batch_size, 32]
        embi = self.m_itemEmb.weight.data                          # [itemCount, 32]

        res = embu @ embi.T                                         # [batch_size, itemCount]
        normU = torch.linalg.norm(embu, dim=1, ord=2)              # [batch_size]
        normI = torch.linalg.norm(embi, dim=1, ord=2)              # [itemCount]

        normU = normU.unsqueeze(dim=1)                             # [batch_size, 1]
        normI = normI.unsqueeze(dim=0)                             # [1, itemCount]

        res = res / (normU @ normI)                                 # [batch_size, itemCount]

        return res