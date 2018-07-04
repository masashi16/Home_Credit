import construct_data
import prediction
from time import time


def main():

    start = time()

    # データ構築
    X_train, y_train, X_test = construct_data.construct_data()

    # 予測
    prediction.predict(X_train, y_train, X_test, 'submission')

    print('time: %.2f s'%(time()-start))


if __name__ == '__main__':
    main()
