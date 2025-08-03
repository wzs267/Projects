
const router = require('koa-router')()
const controller = require('../controller/api.js')
const {verifyToken} = require('../utils/token.js')
const path = require('path')

const multer = require('@koa/multer')
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null,path.join(__dirname, '../public/uploads')); // 文件存储的目录
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname +
        "-" +
        uniqueSuffix +
        "." +
        file.originalname.split(".").pop()
    ); // 文件名设置
  },
});
const upload = multer({ storage: storage })

// 1.轮播图获取
router.get('/getSwiper', controller.getSwiper)
// 2 推荐歌单 -> 改为AI推荐单曲
router.get('/getRecommendSongList', controller.getSongList)
// 2.1 AI推荐单曲（新接口）
router.get('/getRecommendSongs', controller.getRecommendSongs)
// 3 推荐歌手
router.get('/getRecommendSinger', controller.getSingerList)

// 4 歌手详情通过歌手Id
router.get('/getSingerDetail', controller.getSingerDetail)
// 5 歌单详情通过歌单Id 
router.get('/getListSongDetail', controller.getListSongDetail)

// 6 歌单所有列表
router.get('/getAllSongList', controller.getAllSongList)
// 7. 歌手所有列表
router.get('/getAllSinger', controller.getAllSinger)
// 8 搜索
router.get('/searchSong', controller.searchSong)
// 9. 歌曲详情
router.get('/getSongDetail', controller.getSongDetail)


// 10登录
router.post('/login',controller.login)
// 11 我的喜欢
//getMySongList
router.get('/getMySongList',verifyToken(),controller.getMySongList)


// 12.评论列表
router.get('/commentList',controller.commentList)
// 13.添加评论或回复
router.post('/addComment',verifyToken(),controller.addComment)

// 14.通过评论ID 获取回复评论列表
router.get('/getReplyComment',verifyToken(),controller.getReplyComment)

// 15 收藏与取消收藏
router.get('/setCollect',verifyToken(),controller.setCollect)
// 16 判断是否收藏
router.get('/isCollect',verifyToken(),controller.isCollect)



// 17 修改用户信息
router.post('/userEdit',verifyToken(),controller.userEdit)
// 18  获得用户信息
router.get('/getUserInfo',verifyToken(),controller.getUserInfo)

// 18  上传 一定POST请求
router.post('/uploads',upload.single('file'),async (ctx,next)=>{
  const file = ctx.request.file
  console.log(file)
  ctx.body = {code:200,message:'上传成功',data:file.filename}
})


module.exports = router
