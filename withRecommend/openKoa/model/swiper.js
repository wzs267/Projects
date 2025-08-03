module.exports = (sequelize, DataTypes) => {
    const swiperModel = sequelize.define("swiper", {
      id: {
        type: DataTypes.INTEGER(10),
        primaryKey: true,
        autoIncrement: true,
      },
      title: {
        type: DataTypes.STRING(50),
        comment: "名称",
      },
      url: {
        type: DataTypes.STRING(100),
        comment: "连接",
      },
      imgurl: {
        type: DataTypes.STRING(300),
        comment: "轮播图像",
      },
      state: {
        type: DataTypes.INTEGER(10),  
        defaultValue: 0,
        comment: "状态",
      },
      order: {
        type: DataTypes.INTEGER(10),
        defaultValue: 0,
        comment: "排序",
      },
    });
  
    return swiperModel;
  };
  