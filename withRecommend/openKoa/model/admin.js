module.exports = (sequelize, DataTypes) => {
    const adminModel = sequelize.define("admin", {
      id: {
        type: DataTypes.INTEGER(10),
        primaryKey: true,
        autoIncrement: true,
      },
      username: {
        type: DataTypes.STRING(50),
        allowNull: false,
        unique: true,
        comment: "用户名",
      },
      password: {
        type: DataTypes.STRING(50),
        allowNull: false,
        comment: "密码"
      }
    });
  
    return adminModel;
  };
  