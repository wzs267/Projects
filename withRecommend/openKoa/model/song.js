module.exports = (sequelize, DataTypes) => {
    const song = sequelize.define("song", {
      id: {
        type: DataTypes.INTEGER(10),
        primaryKey: true,
        autoIncrement: true,
      },
      singer_id: {
        type: DataTypes.INTEGER(10),
      },
      name: {
        type: DataTypes.STRING(100),
        
      },
      introduction: {
        type: DataTypes.STRING(255),
        
      },
      pic: {
        type: DataTypes.STRING(255),
        
      },
      lyric: {
        type: DataTypes.TEXT(), //text类型
        
      },
      url: {
        type: DataTypes.STRING(255),
        
      },
    });  
    return song;
  };
  