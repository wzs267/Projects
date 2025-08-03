module.exports = (sequelize, DataTypes) => {
    const comment = sequelize.define("comment", {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      user_id: {
        type: DataTypes.INTEGER,
      },
      type: {
        type: DataTypes.INTEGER,
        
      },
      song_id: {
        type: DataTypes.INTEGER,
        
      },
      song_list_id: {
        type: DataTypes.INTEGER,
        
      },
      content: {
        type: DataTypes.STRING,
  
      },
      up:{
        type: DataTypes.INTEGER,
      }
    });
    return comment;
  };
  