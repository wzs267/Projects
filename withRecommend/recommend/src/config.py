# mymusic数据库配置
class MyMusicConfig:
    # 数据库配置
    DATABASE_NAME = "mymusic"
    
    # 数据表配置
    USERS_TABLE = "consumers"
    SONGS_TABLE = "songs" 
    PLAYS_TABLE = "user_plays"
    
    # 数据范围
    NUM_USERS = 102  # 包含现有的2个用户 + 100个新用户
    NUM_SONGS = 10   # 现有的10首歌曲
    
    # 模型参数 (与原配置保持一致)
    EMBEDDING_DIM = 64
    LSTM_UNITS = 128
    DENSE_UNITS = 64
    USER_TOWER_OUTPUT_DIM = 64
    SONG_TOWER_OUTPUT_DIM = 64
    
    # 训练参数
    EPOCHS = 10
    BATCH_SIZE = 32
    
    # 路径配置
    DATA_PATH = "data/mymusic_generated/user_plays.json"
    PROCESSED_PATH = "data/mymusic_processed/interactions.npy"
    MAPPINGS_PATH = "data/mymusic_processed/mappings.npy"
    MODEL_SAVE_PATH = "models/mymusic_twin_tower.keras"
