module.exports = (sequelize, DataTypes) => {
    const song_list = sequelize.define("song_list", {
      id: {
        type: DataTypes.INTEGER(10),
        primaryKey: true,
        autoIncrement: true,
      },
      // title,pic,introduction,style
      title: {
        type: DataTypes.STRING(255),
        defaultValue: "",
      },
      pic: {
        type: DataTypes.STRING(255),
        defaultValue: "",
      },
      introduction: {
        type: DataTypes.STRING(255),
        defaultValue: "",
      },
      style: {
        type: DataTypes.STRING(255), //流行,摇滚,民谣,电子,古典,爵士,乡村,嘻哈,拉丁,轻音乐,另类/独立,金属,蓝调,雷鬼,世界音乐,拉丁,舞曲,乡村,民族,新世纪,后摇,R&B/Soul,电子乐,流行
      }
    });
    return song_list;
  };
  