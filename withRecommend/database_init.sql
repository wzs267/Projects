/*
 Navicat Premium Data Transfer

 Source Server         : localhost_3306
 Source Server Type    : MySQL
 Source Server Version : 80300 (8.3.0)
 Source Host           : localhost:3306
 Source Schema         : music

 Target Server Type    : MySQL
 Target Server Version : 80300 (8.3.0)
 File Encoding         : 65001

 Date: 09/07/2025 04:18:02
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for admins
-- ----------------------------
DROP TABLE IF EXISTS `admins`;
CREATE TABLE `admins` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `password` varchar(50) NOT NULL COMMENT '密码',
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `username_2` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of admins
-- ----------------------------
BEGIN;
INSERT INTO `admins` (`id`, `username`, `password`, `createdAt`, `updatedAt`) VALUES (1, 'admin', 'e10adc3949ba59abbe56e057f20f883e', '2025-07-09 04:11:35', '2025-07-09 04:11:44');
INSERT INTO `admins` (`id`, `username`, `password`, `createdAt`, `updatedAt`) VALUES (2, 'admin1', 'e10adc3949ba59abbe56e057f20f883e', '2025-07-09 04:11:38', '2025-07-09 04:11:47');
COMMIT;

-- ----------------------------
-- Table structure for collects
-- ----------------------------
DROP TABLE IF EXISTS `collects`;
CREATE TABLE `collects` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `song_id` int NOT NULL,
  `song_list_id` int DEFAULT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of collects
-- ----------------------------
BEGIN;
INSERT INTO `collects` (`id`, `user_id`, `song_id`, `song_list_id`, `createdAt`, `updatedAt`) VALUES (1, 2, 1, NULL, '2025-07-06 21:40:23', '2025-07-06 21:40:25');
INSERT INTO `collects` (`id`, `user_id`, `song_id`, `song_list_id`, `createdAt`, `updatedAt`) VALUES (2, 2, 2, NULL, '2025-07-06 23:22:04', '2025-07-06 23:22:06');
COMMIT;

-- ----------------------------
-- Table structure for comments
-- ----------------------------
DROP TABLE IF EXISTS `comments`;
CREATE TABLE `comments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `type` int NOT NULL,
  `song_id` int DEFAULT NULL,
  `song_list_id` int DEFAULT NULL,
  `content` varchar(255) NOT NULL,
  `up` int NOT NULL,
  `createdAt` datetime NOT NULL,
  `updatedAt` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of comments
-- ----------------------------
BEGIN;
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (1, 2, 1, 1, NULL, '非常好', 0, '2025-07-08 04:30:12', '2025-07-08 04:30:15');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (2, 1, 1, 1, NULL, '赞同', 1, '2025-07-08 04:31:07', '2025-07-08 04:31:10');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (3, 2, 1, 2, NULL, '点赞', 0, '2025-07-08 20:17:04', '2025-07-08 20:17:06');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (4, 2, 1, 2, NULL, 'ok', 3, '2025-07-08 14:30:08', '2025-07-08 14:30:08');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (5, 2, 1, 2, NULL, 'ddd', 1, '2025-07-08 14:44:06', '2025-07-08 14:44:06');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (6, 2, 1, 2, NULL, '1111', 3, '2025-07-08 14:45:56', '2025-07-08 14:45:56');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (7, 2, 1, 2, NULL, 'rrrrr', 1, '2025-07-08 14:50:50', '2025-07-08 14:50:50');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (8, 2, 1, 1, NULL, '2222', 3, '2025-07-08 14:51:31', '2025-07-08 14:51:31');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (9, 2, 1, 1, NULL, '222222', 3, '2025-07-08 14:57:30', '2025-07-08 14:57:30');
INSERT INTO `comments` (`id`, `user_id`, `type`, `song_id`, `song_list_id`, `content`, `up`, `createdAt`, `updatedAt`) VALUES (10, 2, 1, 2, NULL, '33333', 3, '2025-07-08 14:57:51', '2025-07-08 14:57:51');
COMMIT;

-- ----------------------------
-- Table structure for consumers
-- ----------------------------
DROP TABLE IF EXISTS `consumers`;
CREATE TABLE `consumers` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `sex` varchar(255) NOT NULL,
  `phone_num` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `birth` varchar(255) NOT NULL,
  `introduction` varchar(255) NOT NULL,
  `location` varchar(255) NOT NULL,
  `avatar` varchar(255) NOT NULL,
  `create_time` varchar(255) NOT NULL,
  `update_time` varchar(255) NOT NULL,
  `createdAt` datetime NOT NULL,
  `updatedAt` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of consumers
-- ----------------------------
BEGIN;
INSERT INTO `consumers` (`id`, `username`, `password`, `sex`, `phone_num`, `email`, `birth`, `introduction`, `location`, `avatar`, `create_time`, `update_time`, `createdAt`, `updatedAt`) VALUES (1, 'user2', 'e10adc3949ba59abbe56e057f20f883e', '1', '19922584707', '1666193045@qq.com', '2025-03-12 00:00:00', 'sad', '天津', '/img/singerPic/1635182970215liudehua.jpg', '2025-03-12 21:54:01', '2025-05-27 01:15:00', '2025-05-27 01:15:00', '2025-05-27 01:15:00');
INSERT INTO `consumers` (`id`, `username`, `password`, `sex`, `phone_num`, `email`, `birth`, `introduction`, `location`, `avatar`, `create_time`, `update_time`, `createdAt`, `updatedAt`) VALUES (2, 'user1', 'e10adc3949ba59abbe56e057f20f883e', '1', '19922584707', '1666193045@qq.com', '2025-03-12 00:00:00', 'sad', '北京', '/img/singerPic/1635182970215liudehua.jpg', '2025-05-21 23:06:44', '2025-05-21 23:06:44', '2025-05-27 01:15:00', '2025-05-27 01:15:00');
COMMIT;

-- ----------------------------
-- Table structure for list_songs
-- ----------------------------
DROP TABLE IF EXISTS `list_songs`;
CREATE TABLE `list_songs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `song_id` int NOT NULL,
  `song_list_id` int NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of list_songs
-- ----------------------------
BEGIN;
INSERT INTO `list_songs` (`id`, `song_id`, `song_list_id`, `createdAt`, `updatedAt`) VALUES (1, 2, 1, '2025-07-09 04:09:24', '2025-07-09 04:09:39');
INSERT INTO `list_songs` (`id`, `song_id`, `song_list_id`, `createdAt`, `updatedAt`) VALUES (2, 7, 1, '2025-07-09 04:09:28', '2025-07-09 04:09:42');
INSERT INTO `list_songs` (`id`, `song_id`, `song_list_id`, `createdAt`, `updatedAt`) VALUES (3, 8, 1, '2025-07-09 04:09:32', '2025-07-09 04:09:46');
INSERT INTO `list_songs` (`id`, `song_id`, `song_list_id`, `createdAt`, `updatedAt`) VALUES (4, 9, 1, '2025-07-09 04:09:36', NULL);
COMMIT;

-- ----------------------------
-- Table structure for ranks
-- ----------------------------
DROP TABLE IF EXISTS `ranks`;
CREATE TABLE `ranks` (
  `id` int NOT NULL AUTO_INCREMENT,
  `song_list_id` int NOT NULL,
  `consumer_id` int NOT NULL,
  `score` int NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of ranks
-- ----------------------------
BEGIN;
COMMIT;

-- ----------------------------
-- Table structure for singers
-- ----------------------------
DROP TABLE IF EXISTS `singers`;
CREATE TABLE `singers` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(50) NOT NULL,
  `sex` varchar(10) NOT NULL,
  `pic` varchar(255) NOT NULL,
  `birth` varchar(50) NOT NULL,
  `location` varchar(50) NOT NULL,
  `introduction` varchar(255) NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=60 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of singers
-- ----------------------------
BEGIN;
INSERT INTO `singers` (`id`, `name`, `sex`, `pic`, `birth`, `location`, `introduction`, `createdAt`, `updatedAt`) VALUES (1, '刘德华', '1', '/img/singerPic/1635182970215liudehua.jpg', '1961-09-27 00:00:00', '中国香港', '中国知名演员、歌手、词作人、制片人、电影人，影视歌多栖发展的代表艺人之一', '2025-07-09 04:08:22', '2025-07-09 04:08:46');
INSERT INTO `singers` (`id`, `name`, `sex`, `pic`, `birth`, `location`, `introduction`, `createdAt`, `updatedAt`) VALUES (2, '张学友', '1', '/img/singerPic/16351845890412.png', '1968-01-01 00:00:00', '中国香港', '中国香港流行乐男歌手、影视演员、作曲人，毕业于香港崇文英文书院。', '2025-07-09 04:08:26', '2025-07-09 04:08:49');
INSERT INTO `singers` (`id`, `name`, `sex`, `pic`, `birth`, `location`, `introduction`, `createdAt`, `updatedAt`) VALUES (3, '周杰伦', '1', '/img/singerPic/1637422520604src=http___ww2.sinaimg.cn_mw690_001gLqIZly1gvqvur3493j60rn0rndjb02.jpg&refer=http___www.sina.jpg', '1975-06-03 00:00:00', '中国大陆', '周杰伦（Jay Chou），1979年1月18日出生于台湾省新北市，祖籍福建省泉州市永春县，中国台湾流行乐男歌手、音乐人、演员、导演、编剧，毕业于淡江中学', '2025-07-09 04:08:29', '2025-07-09 04:08:54');
INSERT INTO `singers` (`id`, `name`, `sex`, `pic`, `birth`, `location`, `introduction`, `createdAt`, `updatedAt`) VALUES (4, 'Beyond', '2', '/img/singerPic/1637422602085ac6eddc451da81cbf2a1bab95066d0160924317a.jpg', '1975-06-03 00:00:00', '中国香港', 'Beyond，中国香港摇滚乐队，成立于1983年，由黄家驹、黄贯中、黄家强、叶世荣组成。', '2025-07-09 04:08:34', '2025-07-09 04:08:57');
INSERT INTO `singers` (`id`, `name`, `sex`, `pic`, `birth`, `location`, `introduction`, `createdAt`, `updatedAt`) VALUES (5, '孙燕姿', '0', '/img/singerPic/1637422925434src=http___n.sinaimg.cn_sinakd20200609ac_226_w540h486_20200609_4627-iuvaazn5230642.jpg&refer=http___n.sinaimg.jpg', '1975-06-03 00:00:00', '中国大陆', '孙燕姿（Stefanie Sun），1978年7月23日出生于新加坡，祖籍中国广东省潮州市，华语流行乐女歌手，毕业于南洋理工大学。', '2025-07-09 04:08:37', '2025-07-09 04:09:02');
INSERT INTO `singers` (`id`, `name`, `sex`, `pic`, `birth`, `location`, `introduction`, `createdAt`, `updatedAt`) VALUES (6, '邓紫棋', '0', '/img/singerPic/1748271732617Snipaste_2025-05-26_23-01-32.png', '2025-03-19 00:00:00', '中国香港', '港澳地区著名女歌手', '2025-07-09 04:08:41', '2025-07-09 04:09:05');
COMMIT;

-- ----------------------------
-- Table structure for song_lists
-- ----------------------------
DROP TABLE IF EXISTS `song_lists`;
CREATE TABLE `song_lists` (
  `id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(255) NOT NULL DEFAULT '',
  `pic` varchar(255) NOT NULL DEFAULT '',
  `introduction` varchar(255) NOT NULL DEFAULT '',
  `style` varchar(255) DEFAULT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of song_lists
-- ----------------------------
BEGIN;
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (1, '粤语', '/img/songListPic/1635439885111yueyu.jpg', '粤语流行歌曲，一般指香港的用粤语（广东话）唱的流行曲，由于香港的原创作曲家是广东人，又叫广东歌。', '粤语', '2025-07-09 04:06:44', '2025-07-09 04:07:19');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (2, '流行', '/img/songListPic/1637421873123src=http___img.alicdn.com_bao_uploaded_O1CN01mNm5pR1q7ghMb1dRu_!!2-item_pic.png&refer=http___img.alicdn.jpg', '流行歌曲（popular song，pop song，缩写pop），是指那些结构短小、内容通俗、形式活泼、情感真挚，并被广大群众所喜爱，广泛传唱或欣赏，流行一时的甚至流传后世的器乐曲和歌曲。这些乐曲和歌曲，植根于大众生活的丰厚土壤之中。因此，又有“大众音乐”之称。流行音乐的特点是：结构短小、内容通俗、形式活泼、情感真挚．具有很强的时代性。', '流行', '2025-07-09 04:06:49', '2025-07-09 04:07:24');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (3, '说唱', '/img/songListPic/1637421925305src=http___y.qq.com_music_photo_new_T001R300x300M000000YKO1D1es6KE.jpg_max_age=2592000&refer=http___y.qq.jpg', 'rap是一个黑人俚语中的词语，相当于“谈话”（talking），中文意思为说唱，是指有节奏地说话的特殊唱歌形式。发源于纽约贫困黑人聚居区。它以在机械的节奏声的背景下，快速地诉说一连串押韵的诗句为特征。', '说唱', '2025-07-09 04:06:52', '2025-07-09 04:07:28');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (4, '摇滚', '/img/songListPic/1637422081780src=http___spider.ws.126.net_5a5e4d93da248da8245afdcaa4d19121.jpeg&refer=http___spider.ws.126.jpg', '摇滚（Rock and Roll）是一种音乐类型，起源于20世纪40年代末期的美国，20世纪50年代早期开始流行，迅速风靡全球。摇滚乐以其灵活大胆的表现形式和富有激情的音乐节奏表达情感，受到了全世界大多数人的喜爱，并在1960年和1970年形成一股热潮。', '摇滚', '2025-07-09 04:06:56', '2025-07-09 04:07:33');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (5, '纯音乐', '/img/songListPic/1637422139576src=http___imagev2.xmcdn.com_group88_M07_04_2A_wKg5DF95GjPglPxHAAEI_ayOTYc828.JPG&refer=http___imagev2.xmcdn.jpg', '纯音乐，是作曲初衷就不包含填词的音乐。这种音乐模式完全以纯粹优美的音乐来表达作者的情感，所以一般被称为纯音乐。它没有歌词，却能以自己优美的曲调体现美妙的意境。', '纯音乐', '2025-07-09 04:07:00', '2025-07-09 04:07:37');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (6, '中国风', '/img/songListPic/1637422244747src=http___img.redocn.com_sheji_20190717_gudianshuimozhongguofengyinlehaibao_10469135.jpg&refer=http___img.redocn.jpg', '中国风音乐介绍 中国风音乐app是中国电信爱音乐快速发布平台的明星应用产品，爱音乐旗下还有哎姆DJ、哎姆乐拍、哎姆明星脸等系列客户端。用最华丽的词语都无法言喻中国音乐特有的吟唱，最迷人的花朵都不及中国音乐的魅力闪亮！让你每一个音乐细胞都随着中国情调的诱动，随着欢快的音符节奏舞动起来。', '中国风', '2025-07-09 04:07:04', '2025-07-09 04:07:40');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (7, '乐器', '/img/songListPic/1637422292728src=http___file.wantu8.com_wan005_im_a9399.jpg&refer=http___file.wantu8.jpg', '乐器，英文：musical instruments，泛指可以用各种方法奏出音色音律的器物。一般分为民族乐器与西洋乐器。', '乐器', '2025-07-09 04:07:08', '2025-07-09 04:07:45');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (8, '经典歌曲', '/img/songListPic/1637422376140src=http___cbu01.alicdn.com_img_ibank_2018_199_941_9309149991_657527067.310x310.jpg&refer=http___cbu01.alicdn.jpg', '以前的优秀歌曲流传至今，广为流传', '经典歌曲', '2025-07-09 04:07:11', '2025-07-09 04:07:48');
INSERT INTO `song_lists` (`id`, `title`, `pic`, `introduction`, `style`, `createdAt`, `updatedAt`) VALUES (9, 'rap', '/img/songListPic/123.jpg', '一种新式流行说唱', '说唱', '2025-07-09 04:07:15', '2025-07-09 04:07:52');
COMMIT;

-- ----------------------------
-- Table structure for songs
-- ----------------------------
DROP TABLE IF EXISTS `songs`;
CREATE TABLE `songs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `singer_id` int NOT NULL,
  `name` varchar(100) NOT NULL,
  `introduction` varchar(255) NOT NULL,
  `pic` varchar(255) NOT NULL,
  `lyric` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `url` varchar(255) NOT NULL,
  `createdAt` datetime DEFAULT NULL,
  `updatedAt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of songs
-- ----------------------------
BEGIN;
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (1, 1, '刘德华-1情感的禁区(演唱会版)', '情感的禁区', '/img/songPic/tubiao.jpg', '[ti:情感的禁区]\n[ar:刘德华]\n[00:01.00]情感的禁区\n[00:06.00]作词：陈浩贤 作曲：天野滋\n[00:11.00]演唱：刘德华\n[00:28.00]\n[00:30.00]街中飘雨车蓬半\n[00:32.37]开我心湿透水\n[00:36.05]独自飞驰追忆挥不去忧虑\n[00:41.67]当天的我不曾爱惜\n[00:44.33]你痴心暗许\n[00:46.96]常令你独垂泪\n[00:49.19]弄得爱路极崎岖\n[00:52.35]\n[00:53.38]今天的你已跟他去\n[00:56.16]心已被窃取\n[00:59.09]孤单的我只有叹唏嘘\n[01:03.93]踏快车 雨中追\n[01:06.96]但愿停车跟你聚\n[01:09.91]但我知 你的心\n[01:12.68]尽是情感的禁区\n[01:15.09]\n[01:31.36]街灯映照车头\n[01:33.80]撇湿满窗的雨水\n[01:37.86]就象我心头抑郁\n[01:40.32]心中满苦泪\n[01:43.35]车厢中我心神\n[01:45.14]更加仿佛空虚\n[01:48.76]连夜我未能睡\n[01:50.97]内心悔恨如有罪\n[01:53.86]\n[01:55.16]当天的你已消失去\n[01:58.00]心若冷水\n[02:00.78]今天的我只有叹唏嘘\n[02:05.58]愿你知 我空虚\n[02:08.74]但愿重新跟你聚\n[02:11.67]但我知 你的心\n[02:14.45]尽是情感的禁区\n[02:16.64]\n[02:19.63]今天的你已跟他去\n[02:22.35]心已被窃取\n[02:25.23]孤单的我只有叹唏嘘\n[02:29.95]踏快车 雨中追\n[02:33.14]但愿停车跟你聚\n[02:36.07]但我知 你的心\n[02:38.85]尽是情感的禁区\n[02:41.90]愿你知 我空虚\n[02:44.60]但愿重新跟你聚\n[02:47.54]但我知 你的心\n[02:50.32]尽是情感的禁区\n[02:52.66]\n[03:08.43]踏快车 雨中追\n[03:11.86]但愿停车跟你聚\n[03:14.76]但我知 你的心\n[03:17.59]尽是情感的禁区\n[03:20.57]愿你知 我空虚\n[03:23.31]但愿重新跟你聚\n[03:26.24]但我知 你的心\n[03:29.10]尽是情感的禁区\n[03:31.50]', '/song/1635666550874刘德华 - 情感的禁区 (Live).mp3', '2025-07-09 04:05:02', '2025-07-09 04:05:44');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (2, 2, '张学友-1一千个伤心的理由', '一千个伤心的理由', '/img/songPic/tubiao.jpg', '[00:00.00]一千个伤心的理由\n[00:03.55]\n[00:04.39]作词:邢增华\n[00:07.49]作曲:李  菘\n[00:10.68]演唱:张学友\n[00:13.49]\n[00:29.63]爱过的人我已不再拥有\n[00:37.08]许多故事有伤心的理由\n[00:43.04]这一次我的爱情等不到天长地久\n[00:50.35]错过的人是否可以回首\n[00:55.99]\n[00:58.71]爱过的心没有任何请求\n[01:05.55]许多故事有伤心的理由\n[01:11.67]这一次我的爱情等不到天长地久\n[01:18.98]走过的路再也不能停留\n[01:24.12]\n[01:25.88]一千个伤心的理由\n[01:29.39]一千个伤心的理由\n[01:33.00]最後我的爱情在故事里慢慢陈旧\n[01:40.12]一千个伤心的理由\n[01:43.75]一千个伤心的理由\n[01:47.38]最後在别人的故事里我被遗忘\n[02:02.54]\n[02:11.75]爱过的心没有任何请求\n[02:19.17]许多故事有伤心的理由\n[02:25.10]这一次我的爱情等不到天长地久\n[02:32.31]走过的路再也不能停留\n[02:37.69]\n[02:39.17]一千个入伤心的理由\n[02:42.93]一千个伤心的理由\n[02:46.46]最後我的爱情在故事里慢慢陈旧\n[02:53.52]一千个伤心的理由\n[02:57.07]一千个伤心的理由\n[03:00.76]最後在别人的故事里我被遗忘\n[03:10.10]\n[03:33.28]一千个伤心的理由\n[03:36.49]一千个伤心的理由\n[03:40.15]最後我的爱情在故事里慢慢陈旧\n[03:47.21]一千个伤心的理由\n[03:50.84]一千个伤心的理由\n[03:54.37]最後在别人的故事里我被遗忘\n[04:11.13]\n[04:20.05]---END---', '/song/1635666865727张学友 - 一千个伤心的理由.mp3', '2025-07-09 04:05:06', '2025-07-09 04:05:49');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (3, 2, '黎明-真爱在明天', '真爱在明天', '/img/songPic/tubiao.jpg', '[ti:真爱在明天]\n[ar:黎明]\n[00:05.00]作曲:黎沸挥填词:向雪怀\n[00:10.00]\n[00:17.36]（女）只有真爱只有真爱可以一生一世地眷恋\n[00:24.64]所有的爱都会疏远只有真心相爱无论隔多远\n[00:35.28]梦中总有你在旁边不必相见\n[00:43.80]（男）紧闭双眼都会听见飘散於风中一片玩笑声\n[00:50.94]不要想你偏会想你一切的温馨感觉还像似冬天\n[01:01.29]在两颗心里有着明天一生不变\n[02:37.77][01:11.05]（女）从前情未了何来缘尽了\n[02:41.47][01:14.66]隔世见面时亦苦笑\n[02:44.74][01:18.37]（男）为何情是债为何缘是债\n[02:48.21][01:22.01]世世也未还清款款痴心债\n[03:18.85][02:54.15][01:27.80]（合）尝尽了辛辛苦苦的爱都是自愿\n[03:25.92][03:01.17][01:34.78]谁又怕生生死死只要好梦实现\n[03:32.97][03:08.17][01:41.86]情未了心中丝丝真爱不断\n[03:37.00][03:11.84][01:45.71]就算是不相见\n[02:10.04]（女）只有真爱只有真爱可以一生一世未变迁\n[02:17.24]（男）所有的爱（所有的爱）都会转变（都会转变）\n[02:21.00]只有倾心的爱常令我温暖\n[02:27.85]就算经千百个明天始终不变', '/song/1635667191428周慧敏 _ 黎明 - 真爱在明天.mp3', '2025-07-09 04:05:11', '2025-07-09 04:05:54');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (4, 2, '郭富城-Para Para Sakura', '巴拉巴拉', '/img/songPic/tubiao.jpg', '[00:01.99]Para Para Sakura\n[00:03.54]作词：陈少琪 作曲：金培远\n[00:05.49]演唱：郭富城\n[00:07.50]\n[00:29.44]如樱花 盛开到落下\n[00:32.19]像尾音短促你有感觉吗\n[00:35.66]如嘴巴外观吻合吧\n[00:38.45]合上它 合作吗 二进一 愿意吗\n[00:42.06](如卡通的主角摆动吧)\n[00:45.15]若我俩有了爱 关节都软化\n[00:48.36](如指针般手舞足动吧)\n[00:51.49]旋转高中低 左右变出交叉\n[00:54.67]mi ni ko i sakura ah e oh\n[00:58.04]come and dance with me\n[01:00.76]mi ni ko i sakura ah e oh\n[01:04.38]come and dance with me\n[01:07.39]乖乖龙地冬 乖乖龙地冬\n[01:13.80]乖乖龙地冬 \n[01:16.65]come and dance with me\n[01:20.40]如樱花 熬不过仲夏\n[01:23.09]热爱捉不紧 快要起变化\n[01:26.75]如身体 像冰冷大厦\n[01:29.37]没抱拥 便暖吗 没有声愉快吗\n[01:33.07](如卡通的主角摆动吧)\n[01:36.02]若我俩有了爱 关节都软化\n[01:39.21](如指针般手舞足动吧)\n[01:42.35]旋转高中低 最後变出烟花\n[01:45.65]mi ni ko i sakura ah e oh\n[01:48.88]come and dance with me\n[01:51.80]mi ni ko i sakura ah e oh\n[01:55.26]come and dance with me\n[01:58.19]乖乖龙地冬 乖乖龙地冬\n[02:04.67]乖乖龙地冬 \n[02:08.03]come and dance with me\n[02:11.07]\n[02:28.00]mi ni ko i sakura ah e oh\n[02:31.80]come and dance with me\n[02:34.74]mi ni ko i sakura ah e oh\n[02:38.13]come and dance with me\n[02:41.10]乖乖龙地冬 乖乖龙地冬\n[02:47.59]乖乖龙地冬 \n[02:50.81]come and dance with me\n[02:54.01]mi ni ko i sakura ah e oh\n[02:57.25]come and dance with me\n[03:00.33]mi ni ko i sakura ah e oh\n[03:03.61]come and dance with me\n[03:06.65]mi ni ko i sakura ah e oh\n[03:09.89]come and dance with me\n[03:13.05]mi ni ko i sakura ah e oh\n[03:16.28]come and dance with me\n[03:19.46]come and dance with me\n[03:22.54]come and dance with me\n[03:25.78]come and dance with me\n[03:28.97]come and dance with me\n[03:32.04]come and dance with me\n[03:34.15]', '/song/1635667283815郭富城 - Para Para Sakura (国语).mp3', '2025-07-09 04:05:15', '2025-07-09 04:05:58');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (5, 6, '邓紫棋-泡沫', '泡沫', '/img/songPic/tubiao.jpg', '[00:00:00]暂无歌词', '/song/1742366675677M800001ziKgJ3o5Ipp.mp3', '2025-07-09 04:05:19', '2025-07-09 04:06:02');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (6, 3, '周杰伦-告白气球', '周杰伦的床边故事', '/img/songPic/tubiao.jpg', '[00:00.00]告白气球 - 周杰伦 (Jay Chou)\n[00:05.89]\n[00:05.89]作词: 方文山\n[00:11.79]作曲: 周杰伦\n[00:17.69]编曲: 林迈可\n[00:23.59]\n[00:23.59]塞纳河畔 左岸的咖啡\n[00:26.16]我手一杯 品尝你的美\n[00:29.33]留下唇印的嘴\n[00:34.27]\n[00:34.27]花店玫瑰 名字写错谁\n[00:36.90]告白气球 风吹到对街\n[00:40.01]微笑在天上飞\n[00:44.01]\n[00:44.01]你说你有点难追\n[00:46.57]想让我知难而退\n[00:49.22]礼物不需挑最贵\n[00:51.89]只要香榭的落叶\n[00:54.56]\n[00:54.56]喔 营造浪漫的约会\n[00:57.26]不害怕搞砸一切\n[00:59.93]拥有你就拥有全世界\n[01:05.01]\n[01:05.01]亲爱的 爱上你\n[01:08.17]从那天起\n[01:11.33]甜蜜的很轻易\n[01:15.69]\n[01:15.69]亲爱的 别任性\n[01:18.85]你的眼睛\n[01:21.94]在说我愿意\n[01:48.90]\n[01:48.90]塞纳河畔 左岸的咖啡\n[01:51.46]我手一杯 品尝你的美\n[01:54.43]留下唇印的嘴\n[01:59.56]\n[01:59.56]花店玫瑰 名字写错谁\n[02:02.14]告白气球 风吹到对街\n[02:05.23]微笑在天上飞\n[02:09.29]\n[02:09.29]你说你有点难追\n[02:11.90]想让我知难而退\n[02:14.60]礼物不需挑最贵\n[02:17.26]只要香榭的落叶\n[02:19.93]\n[02:19.93]喔 营造浪漫的约会\n[02:22.65]不害怕搞砸一切\n[02:25.27]拥有你就拥有 全世界\n[02:30.31]\n[02:30.31]亲爱的 爱上你\n[02:33.58]从那天起\n[02:36.60]甜蜜的很轻易\n[02:40.94]\n[02:40.94]亲爱的 别任性\n[02:44.20]你的眼睛\n[02:47.26]在说我愿意\n[02:51.76]\n[02:51.76]亲爱的 爱上你\n[02:55.05]恋爱日记\n[02:57.93]飘香水的回忆\n[03:02.33]\n[03:02.33]一整瓶 的梦境\n[03:05.42]全都有你\n[03:08.64]搅拌在一起\n[03:13.02]\n[03:13.02]亲爱的别任性\n[03:16.23]你的眼睛\n[03:21.31]在说我愿意\n[03:25.00]\n[03:25.00]---END---', '/song/1748269955603gaobaiqiqiu.mp3', '2025-07-09 04:05:23', '2025-07-09 04:06:07');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (7, 4, 'Beyond-Beyon-灰色轨迹', '天若有情', '/img/songPic/tubiao.jpg', '[00:00.00]灰色轨迹 - BEYOND\r\n[00:05.46]\r\n[00:05.46]作词: 刘卓辉\r\n[00:10.92]作曲: 黄家驹\r\n[00:16.38]\r\n[00:16.38]酒一再沉溺 何时麻醉我抑郁\r\n[00:23.34]过去了的一切会平息\r\n[00:29.38]\r\n[00:29.38]冲不破墙壁 前路没法看得清\r\n[00:36.51]再有那些挣扎与被迫\r\n[00:43.25]\r\n[00:43.25]踏着灰色的轨迹\r\n[00:50.35]尽是深渊的水影\r\n[00:56.61]\r\n[00:56.61]我已背上一身苦困后悔与唏嘘\r\n[01:03.47]你眼里却此刻充满泪\r\n[01:09.96]这个世界已不知不觉的空虚\r\n[01:15.84]Woo 不想你别去\r\n[01:23.02]\r\n[01:23.02]心一再回忆 谁能为我去掩饰\r\n[01:30.14]到哪里都跟你要认识\r\n[01:36.31]\r\n[01:36.31]洗不去痕迹 何妨面对要可惜\r\n[01:43.40]各有各的方向与目的\r\n[01:50.40]\r\n[01:50.40]踏着灰色的轨迹\r\n[01:57.31]尽是深渊的水影\r\n[02:03.54]\r\n[02:03.54]我已背上一身苦困后悔与唏嘘\r\n[02:10.15]你眼里却此刻充满泪\r\n[02:16.71]这个世界已不知不觉的空虚\r\n[02:22.78]Woo 不想你别去\r\n[02:57.18]\r\n[02:57.18]踏着灰色的轨迹\r\n[03:03.45]尽是深渊的水影\r\n[03:10.16]\r\n[03:10.16]我已背上一身苦困后悔与唏嘘\r\n[03:16.71]你眼里却此刻充满泪\r\n[03:23.15]这个世界已不知不觉的空虚\r\n[03:29.46]Woo 不想你别去\r\n[03:36.90]\r\n[03:36.90]我已背上一身苦困后悔与唏嘘\r\n[03:43.38]你眼里却此刻充满泪\r\n[03:50.12]这个世界已不知不觉的空虚\r\n[03:55.97]Woo 不想你别去\r\n[04:05.00]\r\n[04:05.00]---END---', '/song/1748271984098huiseguiji.mp3', '2025-07-09 04:05:27', '2025-07-09 04:06:11');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (8, 4, 'Beyond-1灰色轨迹', '灰色轨迹', '/img/songPic/tubiao.jpg', '[00:00.00]灰色轨迹 - BEYOND\r\n[00:05.46]\r\n[00:05.46]作词: 刘卓辉\r\n[00:10.92]作曲: 黄家驹\r\n[00:16.38]\r\n[00:16.38]酒一再沉溺 何时麻醉我抑郁\r\n[00:23.34]过去了的一切会平息\r\n[00:29.38]\r\n[00:29.38]冲不破墙壁 前路没法看得清\r\n[00:36.51]再有那些挣扎与被迫\r\n[00:43.25]\r\n[00:43.25]踏着灰色的轨迹\r\n[00:50.35]尽是深渊的水影\r\n[00:56.61]\r\n[00:56.61]我已背上一身苦困后悔与唏嘘\r\n[01:03.47]你眼里却此刻充满泪\r\n[01:09.96]这个世界已不知不觉的空虚\r\n[01:15.84]Woo 不想你别去\r\n[01:23.02]\r\n[01:23.02]心一再回忆 谁能为我去掩饰\r\n[01:30.14]到哪里都跟你要认识\r\n[01:36.31]\r\n[01:36.31]洗不去痕迹 何妨面对要可惜\r\n[01:43.40]各有各的方向与目的\r\n[01:50.40]\r\n[01:50.40]踏着灰色的轨迹\r\n[01:57.31]尽是深渊的水影\r\n[02:03.54]\r\n[02:03.54]我已背上一身苦困后悔与唏嘘\r\n[02:10.15]你眼里却此刻充满泪\r\n[02:16.71]这个世界已不知不觉的空虚\r\n[02:22.78]Woo 不想你别去\r\n[02:57.18]\r\n[02:57.18]踏着灰色的轨迹\r\n[03:03.45]尽是深渊的水影\r\n[03:10.16]\r\n[03:10.16]我已背上一身苦困后悔与唏嘘\r\n[03:16.71]你眼里却此刻充满泪\r\n[03:23.15]这个世界已不知不觉的空虚\r\n[03:29.46]Woo 不想你别去\r\n[03:36.90]\r\n[03:36.90]我已背上一身苦困后悔与唏嘘\r\n[03:43.38]你眼里却此刻充满泪\r\n[03:50.12]这个世界已不知不觉的空虚\r\n[03:55.97]Woo 不想你别去\r\n[04:05.00]\r\n[04:05.00]---END---', '/song/1748272107429huiseguiji.mp3', '2025-07-09 04:05:32', '2025-07-09 04:06:15');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (9, 2, '张学友-饿狼传说', '饿狼传说', '/img/songPic/tubiao.jpg', '[00:00.00]饿狼传说 - 张学友 (Jacky Cheung)\r\n[00:12.00]\r\n[00:12.00]作词: 潘伟源\r\n[00:24.01]作曲: John Laudon\r\n[00:36.02]\r\n[00:36.02]她熄掉晚灯 幽幽掩两肩\r\n[00:39.53]交织了火花 拘禁在沉淀\r\n[00:43.50]心刚被割损 经不起变迁\r\n[00:47.28]她偏以指尖 牵引着磁电\r\n[00:51.74]\r\n[00:51.74]汹涌的爱扑着我尽力乱吻乱缠\r\n[00:59.68]偏偏知道爱令我无明天\r\n[01:06.99]\r\n[01:06.99]她倚着我肩 呼吸响耳边\r\n[01:10.53]高温已产生 色相令人乱\r\n[01:14.64]君子在扑火 吹不走暖烟\r\n[01:18.35]她加上嘴巴 给我做磨练\r\n[01:22.56]\r\n[01:22.56]汹涌的爱扑着我尽力乱吻乱缠\r\n[01:29.85]偏偏知道爱令我无明天\r\n[01:37.88]\r\n[01:37.88]爱会像头饿狼 嘴巴似极甜\r\n[01:41.41]假使走近玩玩她凶相便呈现\r\n[01:45.33]爱会像头饿狼 岂可抱着眠\r\n[01:49.23]她必给我狠狠的伤势做留念\r\n[02:00.19]\r\n[02:00.19]她倚着我肩 呼吸响耳边\r\n[02:03.74]高温已产生 色相令人乱\r\n[02:07.88]君子在扑火 吹不走暖烟\r\n[02:11.53]她加上嘴巴 给我做磨练\r\n[02:15.87]\r\n[02:15.87]汹涌的爱扑着我尽力乱吻乱缠\r\n[02:22.90]偏偏知道爱令我无明天\r\n[02:31.17]\r\n[02:31.17]爱会像头饿狼 嘴巴似极甜\r\n[02:34.64]假使走近玩玩她凶相便呈现\r\n[02:38.51]爱会像头饿狼 岂可抱着眠\r\n[02:42.32]她必给我狠狠的伤势做留念\r\n[02:46.52]\r\n[02:46.52]爱会像头饿狼 嘴巴似极甜\r\n[02:50.14]假使走近玩玩她凶相便呈现\r\n[02:54.15]爱会像头饿狼 岂可抱着眠\r\n[02:57.84]她必给我狠狠的伤势做留念\r\n[03:36.01]\r\n[03:36.01]爱会像头饿狼 嘴巴似极甜\r\n[03:39.49]假使走近玩玩她凶相便呈现\r\n[03:43.51]爱会像头饿狼 岂可抱着眠\r\n[03:47.23]她必给我狠狠的伤势做留念\r\n[03:51.26]\r\n[03:51.26]爱会像头饿狼 嘴巴似极甜\r\n[03:54.93]假使走近玩玩她凶相便呈现\r\n[03:58.93]爱会像头饿狼 岂可抱着眠\r\n[04:02.72]她必给我狠狠的伤势做留念\r\n[04:10.00]\r\n[04:10.00]---END---', '/song/1748278442582elangchuanshuo.mp3', '2025-07-09 04:05:36', '2025-07-09 04:06:19');
INSERT INTO `songs` (`id`, `singer_id`, `name`, `introduction`, `pic`, `lyric`, `url`, `createdAt`, `updatedAt`) VALUES (10, 4, 'Beyond-不再犹豫', '不再犹豫', '/img/songPic/tubiao.jpg', '[00:00.00]不再犹豫 - BEYOND\r\n[00:07.35]\r\n[00:07.35]作词: 梁美薇\r\n[00:14.71]作曲: 黄家驹\r\n[00:22.07]编曲: BEYOND\r\n[00:29.43]\r\n[00:29.43]无聊望见了犹豫\r\n[00:32.99]达到理想不太易\r\n[00:36.59]即使有信心\r\n[00:38.78]斗志却抑止\r\n[00:43.72]\r\n[00:43.72]谁人定我去或留\r\n[00:47.34]定我心中的宇宙\r\n[00:50.97]只想靠两手\r\n[00:53.21]向理想挥手\r\n[00:58.31]\r\n[00:58.31]问句天几高\r\n[00:59.85]心中志比天更高\r\n[01:05.26]自信打不死的心态活到老\r\n[01:11.97]Woo ho\r\n[01:16.02]\r\n[01:16.02]我有我心底故事\r\n[01:19.60]亲手写上每段\r\n[01:21.88]得失乐与悲与梦儿\r\n[01:26.21]Woo ho\r\n[01:30.41]\r\n[01:30.41]纵有创伤不退避\r\n[01:34.01]梦想有日达成\r\n[01:36.22]找到心底梦想的世界\r\n[01:40.72]终可见\r\n[02:09.75]\r\n[02:09.75]谁人没试过犹豫\r\n[02:13.39]达到理想不太易\r\n[02:16.94]即使有信心\r\n[02:19.20]斗志却抑止\r\n[02:24.14]\r\n[02:24.14]谁人定我去或留\r\n[02:27.70]定我心中的宇宙\r\n[02:31.34]只想靠两手\r\n[02:33.60]向理想挥手\r\n[02:38.50]\r\n[02:38.50]问句天几高\r\n[02:40.26]心中志比天更高\r\n[02:45.70]自信打不死的心态活到老\r\n[02:52.51]Woo ho\r\n[02:56.50]\r\n[02:56.50]我有我心底故事\r\n[03:00.04]亲手写上每段\r\n[03:02.36]得失乐与悲与梦儿\r\n[03:06.81]Woo ho\r\n[03:10.81]\r\n[03:10.81]纵有创伤不退避\r\n[03:14.40]梦想有日达成\r\n[03:16.64]找到心底梦想的世界\r\n[03:21.16]终可见\r\n[03:24.70]Woo ho\r\n[03:28.76]\r\n[03:28.76]亲手写上每段\r\n[03:31.00]得失乐与悲与梦儿\r\n[03:39.07]Woo ho\r\n[03:43.12]\r\n[03:43.12]梦想有日达成\r\n[03:45.36]找到心底梦想的世界\r\n[03:49.80]终可见 嘿\r\n[03:55.00]\r\n[03:55.00]---END---', '/song/1748278993234不再犹豫.mp3', '2025-07-09 04:05:40', '2025-07-09 04:06:23');
COMMIT;

-- ----------------------------
-- Table structure for swipers
-- ----------------------------
DROP TABLE IF EXISTS `swipers`;
CREATE TABLE `swipers` (
  `id` int NOT NULL AUTO_INCREMENT,
  `title` varchar(50) NOT NULL COMMENT '名称',
  `url` varchar(100) NOT NULL COMMENT '连接',
  `imgurl` varchar(300) NOT NULL COMMENT '轮播图像',
  `state` int NOT NULL DEFAULT '0' COMMENT '状态',
  `order` int NOT NULL DEFAULT '0' COMMENT '排序',
  `createdAt` datetime NOT NULL,
  `updatedAt` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Records of swipers
-- ----------------------------
BEGIN;
INSERT INTO `swipers` (`id`, `title`, `url`, `imgurl`, `state`, `order`, `createdAt`, `updatedAt`) VALUES (1, '图1', 'https://www.baidu.com', 'focus1.jpg', 0, 0, '2025-07-03 05:21:23', '2025-07-03 05:21:25');
INSERT INTO `swipers` (`id`, `title`, `url`, `imgurl`, `state`, `order`, `createdAt`, `updatedAt`) VALUES (2, '图2', 'https://www.baidu.com', 'focus2.jpg', 0, 0, '2025-07-03 05:21:55', '2025-07-03 05:21:57');
INSERT INTO `swipers` (`id`, `title`, `url`, `imgurl`, `state`, `order`, `createdAt`, `updatedAt`) VALUES (3, '图3', 'htts://www.baidu.com', 'focus3.jpg', 0, 0, '2025-07-03 05:22:27', '2025-07-03 05:22:30');
COMMIT;

SET FOREIGN_KEY_CHECKS = 1;
