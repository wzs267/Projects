const model = require('../model/index')
const Swiper = model.swiper  // 轮播图表
const SongList = model.song_list // 歌单表
const Song = model.song // 歌曲表
const ListSong= model.list_song // 歌单歌曲表
const Singer = model.singer // 歌手表
const Consumer = model.consumer
const Collect = model.collect
const Comment = model.comment
const Sequelize = require('sequelize')
const Op = Sequelize.Op
const md5 = require('md5')
const {generateToken} = require( '../utils/token')

// 1.获得轮播图
module.exports.getSwiper = async (ctx, next) => {
     const swiper = await Swiper.findAll()
     ctx.body = {code: 200, message:'轮播图成功',data: swiper}
}
// 2. 获得推荐歌单 -> 改为AI推荐单曲
exports.getSongList = async (ctx, next) => {
    // 调用新的推荐单曲API
    await exports.getRecommendSongs(ctx, next);
}
// 3. 获得推荐歌手
exports.getSingerList = async (ctx, next) => {
    const singer = await Singer.findAll({
        limit:6,   // 获取6条数据
        order : [['id', 'asc']]  // 升序
    })
    ctx.body = {code: 200, message:'歌手成功',data: singer}
}

// 4 . 获得歌手详情,下面有对应歌曲
exports.getSingerDetail = async (ctx, next) => {
    const {id}= ctx.request.query
    const singer = await Singer.findOne({
        where: {
            id: id
        },
        include: [{
            model: Song,
            attributes: ['id', 'name', 'url', 'pic','introduction']
        }]
    })
    ctx.body = {code: 200, message:'歌手详情成功',data: singer}
}
// 5. 获得歌单详情
exports.getListSongDetail = async (ctx, next) => {
    // ListSong
    const {id}= ctx.request.query
    const listSong = await ListSong.findAll({
        where: {
            song_list_id: id
        },
        attributes: ['song_id', 'song_list_id'],
        include: [{
            model: Song,
            attributes: ['id', 'name', 'url', 'pic','introduction']
        },{
           model: SongList,
           attributes: ['id', 'title', 'pic','introduction']
        }     
        ],
        raw: true,  // json数据
        nest: true
        
    })
    let  newList = {}
    let songs = []
    if(listSong.length > 0){
        newList.songList = listSong[0].song_list
        listSong.forEach(item => {
            songs.push(item.song)
                
        })
    }
    
    newList.songs = songs
    console.log(newList)
    ctx.body = {code: 200, message:'歌单详情成功',data: newList}
}
// 6 带分页歌单所有列表
exports. getAllSongList = async (ctx) => { 
    const {page=1, pageSize=6} = ctx.query
    const songList = await SongList.findAll({
        limit: pageSize,   // 获取6条数据
        offset: (page - 1) * pageSize,  // 跳过多少条数据
        order : [['id', 'asc']]  // 升序
    })
    ctx.body = {code: 200, message:'歌单所有列表成功',data: songList}
}
// 7. 带分页歌手所有列表
exports.getAllSinger = async (ctx) => { 
    const {page=1, pageSize=6} = ctx.query
    const singer = await Singer.findAll({
        limit: pageSize,   // 获取6条数据
        offset: (page - 1) * pageSize,  // 跳过多少条数据
        order : [['id', 'asc']]  // 升序
    })
    ctx.body = {code: 200, message:'歌手所有列表成功',data: singer}
}

// 8 带分页搜索歌曲searchSong
exports.searchSong = async (ctx) => {
    const {keywords,page=1, pageSize=3, type=1} = ctx.query
    let list = null
    if(type == 1){  // 搜索歌曲
        list = await Song.findAll({
            where: {
                name: {
                    [Op.like]: `%${keywords}%`
                }
            },
            limit: pageSize,   // 获取6条数据
            offset: (page - 1) * pageSize,  // 跳过多少条数据
            order : [['id', 'asc']]  // 升序
        })
    } else if(type == 2){  // 搜索歌手
        list = await Singer.findAll({
            where: {
                name: {
                    [Op.like]: `%${keywords}%`
                }
            },
            limit: pageSize,   // 获取6条数据
            offset: (page - 1) * pageSize,  // 跳过多少条数据
            order : [['id', 'asc']]  // 升序
        })
    }else if(type == 3){  // 搜索歌单
        list = await SongList.findAll({
            where: {
                title: {
                    [Op.like]: `%${keywords}%`
                }
            },
            limit: pageSize,   // 获取6条数据
            offset: (page - 1) * pageSize,  // 跳过多少条数据
            order : [['id', 'asc']]  // 升序
        })
    }
    datas = {
        list,
        type
    }
    ctx.body = {code: 200, message:'获取成功',data: datas}
}
// 通过Id ,获取歌曲详情 
exports.getSongDetail = async (ctx) => {
    const {id}= ctx.request.query
    console.log(id)
    const song = await Song.findAll({
        where: {
            id: id
        },
        include: [{
            model: Singer
        }]
    })
    ctx.body = {code: 200, message:'获取成功',data: song}
}

// 登录
exports.login = async (ctx, next) =>{
        const {username, password} = ctx.request.body
        const user = await Consumer.findOne({
            where:{
                username,
                password:md5(password)
            },
            //attributes:['username','avatar']  
        })
        // 存token 返回给客户的
        if(user){
            // 登录成功
            const token = await generateToken({username:user.username,id:user.id})
            ctx.body = {code: 200, message:"登录成功",data:{user,token}}
        }

}
exports.getMySongList= async (ctx, next) =>{
    const userId = ctx.user.id
    //const userId = 2 // 用户固定
    let list = await Collect.findAll({
        where:{user_id:userId},
        attributes:['id'],
        include:[{model:Song},{model:Consumer,attributes:['username','avatar']}]
        
    })

    let  newList = {}
    let songs = []
    if(list.length > 0){
        newList.consumer = list[0].consumer
        list.forEach(item => {
            songs.push(item.song)    
        })
    }
    newList.songs = songs
    ctx.body = {code: 200, message:"我的歌单",data: newList}
}

// 评论列表
exports.commentList= async (ctx, next) =>{
    const include = [{model:Consumer,attributes:['username','avatar',]}]

    let lists = await Comment.findAll({
        include,
        order:[['createdAt','DESC']],
        raw:true,  // 返回的数据是原始的
        nest:true
    })
    console.log(lists)
    let resData = []
    lists.forEach(item=>{
        let reply = lists.filter(data=>data.up==item.id) // 返回每条评论下的回复列表
        if(reply.length>0){
            item.reply = reply
        }
        if(item.up==0){
            resData.push(item)
        }
    })
    resData = await Promise.all(resData.map(async item => {
            // 假设这里直接通过item.song获取歌曲信息，根据实际情况调整
            if(item.song_id){
                // 歌曲
                let song = await Song.findOne({
                where: { id: item.song_id }
                });
                // 点赞数
                let likeCount = await Collect.count({
                    where: { song_id: item.song_id }
                })
                // 评论数
                // 不等于 up
                let commentCount = await Comment.count({
                    where: { song_id: item.song_id, up: { [Op.ne]: 0 } }
                })
                // 歌手
                let singer = await Singer.findOne({
                    where: { id: song.singer_id },
                    attributes: ['name']
                });
                
                return {
                    ...item,
                    song: song,
                    singer: singer.name,
                    commentCount,
                    likeCount

                };
            }
    }));
    ctx.body = {code: 200, message:"评论列表",data: resData}
}
// 添加评论 up=0，回复评论 up不是0 
exports.addComment= async (ctx, next) =>{
   // const userId = 2
   const userId = ctx.user.id
    const {content,type=1,up=0,songId}  = ctx.request.body
    console.log(content,type,up,userId,songId)
    await Comment.create({content,type,up,user_id:userId,song_id:songId})
    ctx.body = {code: 200, message:"添加成功",data: null}
}
// 删除评论
exports.delComment= async (ctx, next) =>{
    const {id} = ctx.query
    await Comment.destroy({where:{id}})
}

// 通过评论ID 获取回复评论列表
exports.getReplyComment= async (ctx, next) =>{
    const {id} = ctx.query
    const include = [{model:Consumer,attributes:['username','avatar',]}]

    let lists = await Comment.findAll({
        where:{up:id},
        include,
        order:[['createdAt','DESC']],
        raw:true,  // 返回的数据是原始的
        nest:true
    })
    ctx.body = {code: 200, message:"回复评论列表",data: lists}
}

// 收藏歌曲
exports.setCollect= async (ctx, next) =>{
    const {id,action} = ctx.query  // action =='add'  'cancel'
    // 带用户信息的操作
    //const userId = 2
    const userId = ctx.user.id
    console.log(userId)
    if(action=='add'){ // 收藏 create
        await Collect.create({song_id:id,user_id:userId})

    }else{ // 取消 destory
        await Collect.destroy({where:{song_id:id,user_id:userId}})
    }
    ctx.body = {code: 0, message:"收藏操作成功"}
}

// 判断是否是收藏或取消收藏，在第一次加载页面必须保持一致
exports.isCollect= async (ctx, next) =>{
    const {ids} = ctx.query
    //const userId = 2
    const userId = ctx.user.id
    let res = await Collect.findAll({
        // in 查询 ids 是数组
        where: {song_id:{[Op.in]: ids.split(',')},user_id:userId}
    })
    //
    res = ids.split(',').map(id=>{
        return res.find(item=>item.song_id == id)?true:false
    })
    
    ctx.body = {code: 0, message:"是否收藏",data:res}
    
}

// 修改用
exports.userEdit= async (ctx, next) =>{
    const userId = ctx.user.id
    let params = ctx.request.body
    params.updatedAt = new Date()
    const res = await Consumer.update(params,{where:{id:userId}})
    if(res[0]==1){
        ctx.body = {code: 0, message:"修改成功"}
    }else{
        ctx.body = {code: 1, message:"修改失败"}
    }

}

// 获得用户信息
exports.getUserInfo= async (ctx, next) =>{
    const userId = ctx.user.id
    let user = await Consumer.findOne({where:{id:userId}})
    ctx.body = {code: 200, message:"用户信息",data:{user}}
}

// 获取AI推荐单曲（基于机器学习模型）
exports.getRecommendSongs = async (ctx, next) => {
    const { userId = 3, count = 6 } = ctx.request.query;
    
    try {
        const { exec } = require('child_process');
        const path = require('path');
        
        // 推荐系统脚本路径 - 修正路径
        const scriptPath = path.join(__dirname, '../../recommend/scripts/api/simple_recommend.py');
        const workingDir = path.join(__dirname, '../../recommend');
        const command = `python "scripts/api/simple_recommend.py" ${userId} ${count}`;
        
        console.log('执行推荐命令:', command);
        console.log('工作目录:', workingDir);
        
        // 调用推荐系统
        const result = await new Promise((resolve, reject) => {
            exec(command, { cwd: workingDir }, (error, stdout, stderr) => {
                console.log('Python脚本输出:', stdout);
                console.log('Python脚本错误输出:', stderr);
                
                // 即使有错误，也尝试解析JSON（因为TensorFlow警告会导致错误代码）
                try {
                    // 解析Python脚本返回的JSON
                    const jsonMatch = stdout.match(/\{.*\}/s);
                    if (jsonMatch) {
                        const jsonResult = JSON.parse(jsonMatch[0]);
                        console.log('解析的JSON结果:', jsonResult);
                        
                        // 检查结果是否成功
                        if (jsonResult.success) {
                            resolve(jsonResult);
                            return;
                        }
                    }
                } catch (parseError) {
                    console.error('JSON解析失败:', parseError);
                }
                
                // 如果到这里说明解析失败或结果不成功
                if (error) {
                    console.error('推荐系统调用失败:', error);
                    reject(error);
                } else {
                    reject(new Error('无法解析推荐结果'));
                }
            });
        });
        
        // 获取推荐歌曲的详细信息
        if (result.success && result.recommendations) {
            const songIds = result.recommendations.map(rec => rec.song_id);
            
            // 从数据库获取歌曲详细信息
            const songs = await Song.findAll({
                where: {
                    id: songIds
                },
                include: [{
                    model: Singer,
                    attributes: ['id', 'name', 'pic']
                }],
                attributes: ['id', 'name', 'url', 'pic', 'introduction']
            });
            
            // 按推荐分数排序并添加推荐分数
            const sortedSongs = result.recommendations.map(rec => {
                const song = songs.find(s => s.id === rec.song_id);
                if (song) {
                    return {
                        ...song.toJSON(),
                        recommendScore: rec.score,
                        reason: `AI推荐 (${(rec.score * 100).toFixed(1)}%匹配)`
                    };
                }
                return null;
            }).filter(song => song !== null);
            
            ctx.body = {
                code: 200, 
                message: 'AI推荐单曲成功',
                data: sortedSongs,
                meta: {
                    userId: userId,
                    count: sortedSongs.length,
                    algorithm: 'twin_tower_neural_network'
                }
            };
        } else {
            throw new Error('推荐系统返回无效结果');
        }
        
    } catch (error) {
        console.error('推荐API错误:', error);
        
        // 降级方案：返回热门歌曲
        try {
            const fallbackSongs = await Song.findAll({
                limit: parseInt(count),
                order: [['id', 'asc']],
                include: [{
                    model: Singer,
                    attributes: ['id', 'name', 'pic']
                }],
                attributes: ['id', 'name', 'url', 'pic', 'introduction']
            });
            
            const formattedSongs = fallbackSongs.map(song => ({
                ...song.toJSON(),
                recommendScore: 0.5,
                reason: '热门推荐'
            }));
            
            ctx.body = {
                code: 200,
                message: '推荐单曲成功（降级方案）',
                data: formattedSongs,
                meta: {
                    userId: userId,
                    count: formattedSongs.length,
                    algorithm: 'fallback_popular'
                }
            };
        } catch (fallbackError) {
            ctx.body = {
                code: 500,
                message: '推荐系统暂时不可用',
                error: error.message
            };
        }
    }
}