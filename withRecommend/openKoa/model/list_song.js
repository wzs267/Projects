module.exports = (sequelize, DataTypes) => {
    const list_song = sequelize.define("list_song", {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true,
      },
      song_id: {
        type: DataTypes.INTEGER,
        
      },
      song_list_id: {
        type: DataTypes.INTEGER,
        
      }
    });
    return list_song;
};