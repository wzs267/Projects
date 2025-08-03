const song = require('./song.js');
module.exports = (sequelize, DataTypes) => {
    const collect = sequelize.define("collect", {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true,
      },
      user_id: {
        type: DataTypes.INTEGER,
      },
      song_id: {
        type: DataTypes.INTEGER
      },
      song_list_id: {
        type: DataTypes.INTEGER
      }
    });
    return collect;
};