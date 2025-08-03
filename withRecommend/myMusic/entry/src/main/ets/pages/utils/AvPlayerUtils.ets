import media from '@ohos.multimedia.media'
import emitter from '@ohos.events.emitter'
import { SongModel } from '../model/AppModel'
import { defaultState, PlayStateType } from '../model/PlayStateModel'
import { EmitEventType } from '../model/paramModel'
import router from '@ohos.router'
import playService  from './PlayService'
/**
 * 播放工具类
 */
export class AvPlayerUtils{
  static  player:  media.AVPlayer
  //记录上下文
  static context: Context
  //记录播放总时长
  static duration: number = 0
  //当前的播放的时长
  static time: number = 0
  //当前歌曲的播放状态
  static isPlay: boolean = false
  //播放模式
  static playMode: 'auto' | 'repeat' | 'random' = 'auto'
  //歌曲名称，作者，图片等等
  //定义歌曲列表
  static playList: SongModel[]= [//new SongModel(1,'刘德华-冰雨',"/img/songPic/tubiao.jpg","/song/bingyu.mp3","冰雨")
  ]
  //当前的播放索引
  static playIndex: number = 0

  // 初始化播放对象，可以在页面显示的时候初始化
  static async  init(){
    //创建播放对象
    AvPlayerUtils.player  = await media.createAVPlayer()  //创建播放器  media.createAVPlayer()
    console.log('+++aaa',AvPlayerUtils.player)
    let tmp: PlayStateType = await playService.getState()
    if (tmp.playList.length>0){
      AvPlayerUtils.playList =tmp.playList
    }
    AvPlayerUtils.player.on('stateChange', async (state)=>{
      switch(state){
        case 'initialized':
          console.log('01,initialized')
          await AvPlayerUtils.player.prepare()
          break;

        case 'prepared':
          console.log('02,prepared')
          await AvPlayerUtils.player.play()
          //更新播放状态
          AvPlayerUtils.isPlay = true
          break

        case 'completed':
          console.log('03,completed')
          if(AvPlayerUtils.playMode =='random' &&  AvPlayerUtils.playList.length > 1){
            //随机非自身
            let index = 0
            do{
              index = Math.floor(Math.random() * AvPlayerUtils.playList.length)
            }while( index == AvPlayerUtils.playIndex)

            AvPlayerUtils.playIndex = index
          }else if(AvPlayerUtils.playMode == 'auto'){ //auto模式
            AvPlayerUtils.playIndex++
            AvPlayerUtils.playIndex =  (AvPlayerUtils.playIndex + AvPlayerUtils.playList.length) % AvPlayerUtils.playList.length
          }
          console.log('3366',AvPlayerUtils.playList[AvPlayerUtils.playIndex])
          AvPlayerUtils.playSingle(AvPlayerUtils.playList[AvPlayerUtils.playIndex])  //重要，播放歌曲
          break
      }
    })

    AvPlayerUtils.player.on('durationUpdate',(duration)=>{
      console.log('04,durationUpdate')
      AvPlayerUtils.duration = duration
      console.log('logData', AvPlayerUtils.duration)
    })

    //  监听设置播放时长
    AvPlayerUtils.player.on('seekDone',async()=>{
      console.log('05,seekDone')
      await AvPlayerUtils.player.play()
      // 更新播放状态
      AvPlayerUtils.isPlay = true
    })

    AvPlayerUtils.player.on('timeUpdate',(time)=>{
      console.log('06,timeUpdate')
      AvPlayerUtils.time = time
      console.log('logData777', AvPlayerUtils.time, '发送数据')
      // 收集要通知的信息
      const playState:PlayStateType = {
        pic:AvPlayerUtils.playList[AvPlayerUtils.playIndex].pic,
        name:AvPlayerUtils.playList[AvPlayerUtils.playIndex].name,
        url: AvPlayerUtils.playList[AvPlayerUtils.playIndex].url,
        playIndex: AvPlayerUtils.playIndex,
        time: AvPlayerUtils.time,
        duration:AvPlayerUtils.duration,
        isPlay:AvPlayerUtils.isPlay,
        playMode:AvPlayerUtils.playMode,
        playList:AvPlayerUtils.playList
      }

      //发布通知
      console.log("111",playState)
      AvPlayerUtils.updateState(playState)
    })

  }

  //单曲指定播放
  static playSingle(song: SongModel){
    // 判断是不是列表中的歌曲
    const isList = AvPlayerUtils.playList.some((item:SongModel)=>item.id === song.id)
    // 如果是歌曲列表中的歌曲
    if(isList){
      // 是不是正在播放的
      if(AvPlayerUtils.player?.url === song.url){
        if(AvPlayerUtils.isPlay) {
          // 重新播放（切换当前播放索引的歌曲）
          AvPlayerUtils.changePlay()
        }else{
          // 恢复播放进度
          AvPlayerUtils.player.seek(AvPlayerUtils.time)
        }

      }else{
        // 切换歌曲(更新播放索引切换歌曲)
        AvPlayerUtils.playIndex = AvPlayerUtils.playList.findIndex((item:SongModel)=>item.id===song.id)
        AvPlayerUtils.changePlay()
      }
    }else{
      // 添加到播放列表中，再进行播放
      AvPlayerUtils.playList.unshift(song)
      console.log('song',song.url)
      AvPlayerUtils.playIndex = 0
      AvPlayerUtils.changePlay() //重要切换播放
      console.log('+++playList',AvPlayerUtils.playList)
    }

  }
  /**
   * 切换播放
   */
  static async  changePlay(){
    //重置播放器状态，才能重新设置url
    await AvPlayerUtils.player.reset()
    //需要重置播放器数据
    AvPlayerUtils.duration = 0
    AvPlayerUtils.time = 0
    console.log(AvPlayerUtils.playList[AvPlayerUtils.playIndex].url)
    AvPlayerUtils.player.url='http://192.168.43.21:3000/uploads'+ AvPlayerUtils.playList[AvPlayerUtils.playIndex].url
  }
  //发布通知，订阅者就可以得到数据  emitter.emit 发布 （重要重要）
  // 更新歌曲信息至页面
  static updateState(playState: PlayStateType) {
    console.log("222",JSON.stringify(playState))
    emitter.emit({
      eventId: EmitEventType.UPDATE_STATE
    }, {
      data: {
        playStateStr: JSON.stringify(playState)
      }
    })
    playService.setState(playState)

  }
  /**
   * 播放暂停
   */
  static pause(){

    AvPlayerUtils.player.pause()
    AvPlayerUtils.isPlay = false
    //更新页面
    const playState:PlayStateType = {
      pic:AvPlayerUtils.playList[AvPlayerUtils.playIndex].pic,
      name:AvPlayerUtils.playList[AvPlayerUtils.playIndex].name,
      url: AvPlayerUtils.playList[AvPlayerUtils.playIndex].url,
      playIndex: AvPlayerUtils.playIndex,
      time:AvPlayerUtils.time,
      duration:AvPlayerUtils.duration,
      isPlay:AvPlayerUtils.isPlay,
      playMode:AvPlayerUtils.playMode,
      playList:AvPlayerUtils.playList
    }
    AvPlayerUtils.updateState(playState)


  }

  /**
   * 下一首
   */
  static next(){

    if(AvPlayerUtils.playList.length === 0){
      return
    }

    //随机模式，歌曲个数大于1
    if(AvPlayerUtils.playMode == 'random' && AvPlayerUtils.playList.length > 1){
      //随机非自身
      let index = 0
      do{
        index = Math.floor(Math.random() * AvPlayerUtils.playList.length)
      }while( index == AvPlayerUtils.playIndex)

      AvPlayerUtils.playIndex = index
    }else{
      AvPlayerUtils.playIndex++
      //超过播放列表
      AvPlayerUtils.playIndex =  (AvPlayerUtils.playIndex + AvPlayerUtils.playList.length ) % AvPlayerUtils.playList.length
    }


    AvPlayerUtils.playSingle(AvPlayerUtils.playList[AvPlayerUtils.playIndex])

  }

  /**
   * 上一首
   */
  static prev(){
    if(AvPlayerUtils.playList.length === 0){
      return
    }

    //随机模式，歌曲个数大于1
    if(AvPlayerUtils.playMode == 'random' && AvPlayerUtils.playList.length > 1){
      //随机非自身
      let index = 0
      do{
        index = Math.floor(Math.random() * AvPlayerUtils.playList.length)
      }while( index == AvPlayerUtils.playIndex)

      AvPlayerUtils.playIndex = index
    }else{
      AvPlayerUtils.playIndex--
      //超过播放列表
      AvPlayerUtils.playIndex =  (AvPlayerUtils.playIndex + AvPlayerUtils.playList.length ) % AvPlayerUtils.playList.length
    }

    AvPlayerUtils.playSingle(AvPlayerUtils.playList[AvPlayerUtils.playIndex])

  }

  /**
   * 切换播放模式
   */
  static changeMode(mode: 'auto' | 'repeat' | 'random'){
    AvPlayerUtils.playMode = mode

    //更新页面
    const playState:PlayStateType = {
      pic:AvPlayerUtils.playList[AvPlayerUtils.playIndex].pic,
      name:AvPlayerUtils.playList[AvPlayerUtils.playIndex].name,
      url: AvPlayerUtils.playList[AvPlayerUtils.playIndex].url,
      playIndex: AvPlayerUtils.playIndex,
      time:AvPlayerUtils.time,
      duration:AvPlayerUtils.duration,
      isPlay:AvPlayerUtils.isPlay,
      playMode:AvPlayerUtils.playMode,
      playList:AvPlayerUtils.playList
    }
    AvPlayerUtils.updateState(playState)

  }


  /**
   * 从播放列表中删除歌曲
   * index:要删除的歌曲的索引,从播放列表中删除,playList中
   */
  static remove(index: number){

    //正在播放的歌曲，将其删除
    if(AvPlayerUtils.playIndex == index){
      //还有其他的歌曲
      if(AvPlayerUtils.playList.length > 1){
        AvPlayerUtils.playList.splice(index,1) //index: 0 ,playIndex=0, len:3
        //切歌
        AvPlayerUtils.playIndex =  (AvPlayerUtils.playIndex + AvPlayerUtils.playList.length ) % AvPlayerUtils.playList.length

        AvPlayerUtils.playSingle(AvPlayerUtils.playList[AvPlayerUtils.playIndex])
      }else{
        //更新一下状态

        AvPlayerUtils.player.reset()
        AvPlayerUtils.playList.splice(index, 1)
        AvPlayerUtils.isPlay = false
        AvPlayerUtils.duration = 0
        AvPlayerUtils.time = 0
        AvPlayerUtils.updateState(defaultState)
        router.back()

      }
    }else{ //删除前面的

      if(AvPlayerUtils.playIndex > index){
        AvPlayerUtils.playIndex--
      }
      AvPlayerUtils.playList.splice(index, 1)

      //发布状态
      AvPlayerUtils.updateState({
        pic: AvPlayerUtils.playList[AvPlayerUtils.playIndex].pic,
        name: AvPlayerUtils.playList[AvPlayerUtils.playIndex].name,
        url: AvPlayerUtils.playList[AvPlayerUtils.playIndex].url,
        playIndex: AvPlayerUtils.playIndex,
        time: AvPlayerUtils.time,
        duration: AvPlayerUtils.duration,
        isPlay: AvPlayerUtils.isPlay,
        playMode: AvPlayerUtils.playMode,
        playList: AvPlayerUtils.playList
      })

    }
  }
}