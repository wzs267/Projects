module.exports = (sequelize, DataTypes) => {
    const consumer = sequelize.define("consumer", {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true,
      },
      username: {
        type: DataTypes.STRING,
        
      },
      password: {
        type: DataTypes.STRING,
        
      },
      sex: {
        type: DataTypes.INTEGER,
       
      },
      phone_num: {
        type: DataTypes.STRING,
        
      },
      email: {
        type: DataTypes.STRING,
        
      },
      birth: {
        type: DataTypes.STRING,
        
      },
      introduction: {
        type: DataTypes.STRING,
       
      },
      location: {
        type: DataTypes.STRING,
       
      },
      avatar: {
        type: DataTypes.STRING,
        
      }
    });
    return consumer;
  };
  