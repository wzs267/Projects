module.exports = (sequelize, DataTypes) => {
    const singer = sequelize.define("singer", {
      id: {
        type: DataTypes.INTEGER(10),
        primaryKey: true,
        autoIncrement: true,
      },
      // name,sex,pic,birth,location,introduction
      name: {
        type: DataTypes.STRING(50),
        
      },
      sex: {
        type: DataTypes.STRING(10),
        
      },
      pic: {
        type: DataTypes.STRING(255),
        
      },
      birth: {
        type: DataTypes.STRING(50),
        
      },
      location: {
        type: DataTypes.STRING(50),
       
      },
      introduction: {
        type: DataTypes.STRING(255),
        
      },
    });
    return singer;
  };