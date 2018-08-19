from scipy import spatial


class Neighbor:

    train_x = None
    train_y = None

    def __init__(self, train_x, train_y):
        """
        初期化関数
        :param train_x: 学習用データセットの入力
        :param train_y: 学習用データセットの出力
        """
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, test_x):
        """
        入力から最近傍法より予測ラベルを返す
        :param test_x: 入力ベクトルのアレイ
        :return pred_y: 出力ラベルのアレイ
        """
        pred_y = []
        for pred_x in test_x:
            best_similarity = -1
            predict_label = None
            for (i, x) in enumerate(self.train_x):
                cos_sim = 1 - spatial.distance.cosine(pred_x, x) #コサイン類似度を距離として計算
                if cos_sim > best_similarity:
                    best_similarity = cos_sim
                    predict_label = self.train_y[i]
            pred_y.append(predict_label)
        return pred_y


