import numpy as np
import re
import os
import pickle
from recommend_system import neighbor
from sklearn.model_selection import KFold


class Recommend:
    genre_path = None
    review_path = None
    user_path = None
    item_path = None
    user_num = None
    genre_num = None
    movie_num = None
    review_num = None
    genre_att = None
    review_att = None
    movie_att = None
    user_att = None
    pickle_path = None

    def __init__(self):
        """
        初期化関数

        """
        self.genre_path = "./ml-100k/u.genre"
        self.review_path = "./ml-100k/u.data"
        self.user_path = "./ml-100k/u.user"
        self.movie_path = "./ml-100k/u.item"
        self.pickle_path = "../dataset_norm.pickle"

        self.genre_att = ["unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
                          "Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
                          "War","Western"]
        self.review_att = ["user_id","movie_id","rating","timestamp"]
        self.movie_att = ["movie_id","movie_title","release_date","video_release_date","IMDb_URL","unknown","Action",
                          "Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy",
                          "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
        self.user_att = ["user_id","age","gender","occupation","zip_code"]

        self.user_num = 943
        self.genre_num = len(self.genre_att)
        self.movie_num = 1682
        self.review_num = 100000


    def load_data(self, path, att, split_char):
        """
        ファイルの読み出し

        :param path: 読み込むデータファイルのパス
        :param att: 読み込むデータの属性名
        :param split_char: データ分割の文字
        :return data_list: データのリスト
        """
        data_list = []
        with open(path, "r", encoding="ISO-8859-1") as f:
            lines = f.readlines()
        for line in lines:
            vec = {}
            for i, item in enumerate(re.split(split_char, line[:-1])):
                vec[att[i]] = item
            data_list.append(vec)
        return data_list

    def merge_data(self, review_list, user_list, movie_list):
        """
        user_idとmovie_idから、各詳細データを取り出し、review情報とマージする
        :param review_list: reviewデータのリスト
        :param user_list: 　userデータのリスト
        :param movie_list:  movieデータのリスト
        :return merged_data_list: user, movieデータを展開して、reviewデータとマージしたリスト
        """
        merged_data_list = []
        for review in review_list:
            user_id = review["user_id"]
            movie_id = review["movie_id"]
            user_data = [user for user in user_list if user["user_id"] == user_id]
            movie_data = [movie for movie in movie_list if movie["movie_id"] == movie_id]
            """
            print(user_id)
            print(movie_id)
            print(user_data)
            print(movie_data)
            """
            user_data[0].update(movie_data[0])
            #print(user_data)
            merged_data = user_data[0].copy()
            merged_data["rating"] = review["rating"]
            merged_data_list.append(merged_data)
        return merged_data_list

    def normalize(self, x):
        """
        ベクトルxの正規化をする
        :param x: 正規化するベクトル
        :return norm_x: 正規化されたベクトル
        """
        norm_x = x / np.linalg.norm(x)
        return norm_x

    def make_movie_vec(self, data_list):
        """
        ユーザごとの映画ベクトル（仮名）を作成する。

        :param data_list: reviewデータのリスト
        :return movie_vec_array: ユーザごとの映画ベクトルのアレイ
        :return recommend_movie_array: ユーザごとのレーティングが最も高い映画idのアレイ
        """
        movie_vec_array = np.empty((0, self.genre_num+1))
        recommend_movie_array = np.empty(0)
        user_id_array = np.empty(0)
        for user_id in range(self.user_num):
            user_id = str(user_id + 1)
            best_movie_rating = 0
            best_movie_id = None
            review_list = [review for review in data_list if review["user_id"] == user_id]
            movie_vec = np.array([0.0 for i in range(self.genre_num)])
            for review in review_list:
                vec = np.empty(0)
                rating = int(review["rating"])
                if best_movie_rating < rating:
                    best_movie_id = review["movie_id"]
                for genre in self.genre_att:
                    vec = np.append(vec, int(review[genre]) * (rating - 3))
                    #vec = np.append(vec, int(review[genre]))
                movie_vec += vec
            user_id_array = np.append(user_id_array, np.array([review_list[0]["user_id"]]), axis=0)
            #movie_vec = np.reshape(np.append(movie_vec, np.array([review_list[0]["age"]]), axis=0), (1, -1))
            movie_vec = np.reshape(np.append(movie_vec, np.array(0)),(1, -1))
            movie_vec = self.normalize(np.asarray(movie_vec, dtype="float32"))
            #movie_vec = np.asarray(movie_vec, dtype="float32")
            movie_vec_array = np.append(movie_vec_array, movie_vec, axis=0)
            recommend_movie_array = np.append(recommend_movie_array, np.array([best_movie_id]), axis=0)
        return movie_vec_array, recommend_movie_array, user_id_array



    def id_to_title(self, movie_id, movie_list):
        """
        映画ＩＤから映画タイトルを返す
        :param movie_id: 映画ＩＤ
        :param movie_list: 映画データのリスト
        :return movie_title: 映画タイトル
        """
        return [movie["movie_title"] for movie in movie_list if movie["movie_id"] == movie_id][0]

    def load_dataset(self):
        """
        pickleデータにあるデータセットを読み込む
        """
        with open(self.pickle_path, "rb") as f:
            obj = pickle.load(f)
        return obj[0], obj[1], obj[2], obj[3]

    def train(self):
        """
        映画のレコメンドモデル(最近傍法)の学習を行う

        """

        if not os.path.exists(self.pickle_path):
            review_list = self.load_data(self.review_path, self.review_att, "	")
            user_list = self.load_data(self.user_path, self.user_att, "\|")
            movie_list = self.load_data(self.movie_path, self.movie_att, "\|")
            print("loaded_data!")

            data_list = self.merge_data(review_list, user_list, movie_list)
            print("merged_data!")
            x_data, y_data, user_list = self.make_movie_vec(data_list)
            print("make_dataset!")
            with open(self.pickle_path, "wb") as f:
                pickle.dump([x_data, y_data, user_list, movie_list], f)
        else:
            x_data, y_data, user_list, movie_list = self.load_dataset()

        n_fold = 100
        k_fold = KFold(n_fold, shuffle=True)

        for train_idx, test_idx in k_fold.split(x_data, y_data):
            train_x = x_data[train_idx]
            test_x = x_data[test_idx]
            train_y = y_data[train_idx]
            test_y = y_data[test_idx]
            user_list[test_idx]

            #knn = KNeighborsClassifier(n_neighbors=1, p=1, metric="minkowski")
            #knn.fit(train_x, train_y)

            #predict_test = knn.predict(test_x)
            nn = neighbor.Neighbor(train_x, train_y)
            predict_test = nn.predict(test_x)
            predict_movie = []
            for movie_id in predict_test:
                predict_movie.append(self.id_to_title(movie_id, movie_list))
            correct_movie = []
            for movie_id in test_y:
                correct_movie.append(self.id_to_title(movie_id, movie_list))
            print(test_y)
            print(predict_test)
            print(correct_movie)
            print(predict_movie)
            break


if __name__ == "__main__":
    model = Recommend()
    model.train()
