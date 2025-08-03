module.exports = (sequelize, DataTypes) => {
    const rank = sequelize.define("rank", {
      id: {
        type: DataTypes.INTEGER(10),
        primaryKey: true,
        autoIncrement: true,
      },
      song_list_id: {
        type: DataTypes.INTEGER(10),
        
      },
      consumer_id: {
        type: DataTypes.INTEGER(10),
        
      },
      score: {
        type: DataTypes.INTEGER(10),
        
      },
    });
    return rank;
  };
  