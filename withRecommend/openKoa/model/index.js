// 1. 连接mysql的配置
const config = require('../config/db.js');
// 2.连接mysql (使用sequelize )
const { Sequelize, DataTypes } = require('sequelize');
const sequelize = new Sequelize(config.database, config.username, config.password,config.option);
// 3. 引入表结构对象
const admin = require('./admin.js')(sequelize,DataTypes);
const collect = require('./collect.js')(sequelize,DataTypes);
const comment = require('./comment.js')(sequelize,DataTypes);
const consumer = require('./consumer.js')(sequelize,DataTypes);
const rank = require('./rank.js')(sequelize,DataTypes);
const singer = require('./singer.js')(sequelize,DataTypes);
const song_list = require('./song_list.js')(sequelize,DataTypes);
const song = require('./song.js')(sequelize,DataTypes);
const swiper = require('./swiper.js')(sequelize,DataTypes);
const list_song = require('./list_song.js')(sequelize,DataTypes);

// singer 和 song 一对多关系 : 一个歌手有多个歌曲，一个歌曲属于一个歌手
singer.hasMany(song, { foreignKey: 'singer_id' });
song.belongsTo(singer, { foreignKey: 'singer_id' });

// song,song_list list_song 歌曲歌单表，多对多关系 : 一个歌单有多个歌曲，一个歌曲属于多个歌单
song.hasMany(list_song, { foreignKey: 'song_id' });
list_song.belongsTo(song, { foreignKey: 'song_id' });

song_list.hasMany(list_song, { foreignKey: 'song_list_id' });
list_song.belongsTo(song_list, { foreignKey: 'song_list_id' });

//  song , consumer ,collect 多对多关系 : 一个用户可以收藏多个歌曲，一个歌曲可以被多个用户收藏
song.hasMany(collect, { foreignKey: 'song_id' });
collect.belongsTo(song,{ foreignKey: 'song_id' });

collect.belongsTo(consumer, { foreignKey: 'user_id' });
consumer.hasMany(collect, { foreignKey: 'user_id' });

// consumer,song comment 多对多关系: 一个用户可以评论多个歌曲，一个歌曲可以被多个用户评论
song.hasMany(comment, { foreignKey: 'song_id' });
comment.belongsTo(song,{ foreignKey: 'song_id' });

comment.belongsTo(consumer, { foreignKey: 'user_id' });
consumer.hasMany(comment, { foreignKey: 'user_id' });

// 5.模型同步   sequelize.sync() 自动同步所有模型
sequelize.sync()
// 6 . 导出模型
exports.admin = admin;
exports.collect = collect;
exports.comment = comment;
exports.consumer = consumer;
exports.rank = rank;
exports.singer = singer;
exports.song_list = song_list;
exports.song = song;
exports.swiper = swiper;
exports.list_song = list_song;
exports.sequelize = sequelize;



