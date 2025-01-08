import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger(__name__)


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def save_data(df, df_traget_path):
    df.to_csv(df_traget_path, index=False)


def target_convert(train):
    train = train.rename(columns={'Survived': 'target'})
    return train


# замена NaN в Age на среднее значение возраста
def clean_null(train, test):
    train['Age'] = train['Age'].fillna(train.Age.mean())
    test['Age'] = test['Age'].fillna(train.Age.mean())
    return train, test


# кодирование данных о поле пассажира числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
def sex_convert(train, test):
    train['Sex'] = train['Sex'].replace(['male', 'female'], [0, 1])
    test['Sex'] = test['Sex'].replace(['male', 'female'], [0, 1])
    return train, test


def one_hot(df, column_names):
    # Создание Объекта OneHotEncoder() и его "обучение" .fit
    ohe = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)
    ohe.fit(df[column_names])

    # Применяем трансформацию .transform и сохраняем результат в Dataframe
    ohe_feat = ohe.transform(df[column_names])
    df_ohe = pd.DataFrame(ohe_feat, columns=ohe.get_feature_names_out()).astype(int)
    return df_ohe


def main(train_df_path, test_df_path, processed_train_path, processed_test_path):
    df_train, df_test = load_data(train_df_path, test_df_path)
    df_train, df_test = clean_null(df_train, df_test)
    df_train, df_test = sex_convert(df_train, df_test)

    df_train = pd.concat([df_train, one_hot(df_train, ['Sex'])], axis=1).reindex(df_train.index)
    df_test = pd.concat([df_test, one_hot(df_test, ['Sex'])], axis=1).reindex(df_test.index)

    df_train = target_convert(df_train)
    df_test = target_convert(df_test)
    save_data(df_train, processed_train_path)
    save_data(df_test, processed_test_path)


if __name__ == "__main__":
    main("./data/titanic_train.csv",
         "./data/titanic_test.csv",
         "./data/processed_train.csv",
         "./data/processed_test.csv")
