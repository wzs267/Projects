import { Home } from './tabs/Home'
import { Comment } from './tabs/Comment'
import { LikeSongList } from './tabs/LikeSongList'
import { Mine } from './tabs/Mine'
import { tabsData, TabBarModel } from './model/TabBar'
import userService from './services/UserService'
import router from '@ohos.router';
import PlayerNav from './widget/playerNav'
import { PlayStateModel, PlayStateType } from './model/PlayStateModel'
import emitter from '@ohos.events.emitter'
import { EmitEventType } from './model/paramModel'
import { AvPlayerUtils } from './utils/AvPlayerUtils';
import CollectService from './services/CollectService'
import { geMySongList } from './common/Api'
import { MySongModel, UserModel, SongCommentModel } from './model/AppModel'
import promptAction from '@ohos.promptAction'
import CommentService from './services/CommentService'

@Entry
@Component
struct Index {
  @State opacityNum: number = 1
  @StorageLink("isLogin") isLogin: boolean = false
  @StorageLink("userInfo") userInfo: UserModel = {}
  @State currentIndex: number = 0
  @State mySongList: MySongModel = new MySongModel({ username: '', avatar: '' }, [])
  @State isCollectList: boolean[] = []
  @State momentList: SongCommentModel[] = []
  // 初始化
  @State
  playState: PlayStateType = new PlayStateModel({
    "pic": "/img/songPic/tubiao.jpg",
    "name": "音乐网",
    "url": "",
    "playIndex": 0,
    "time": 0,
    "duration": 0,
    "isPlay": false,
    "playMode": "auto",
    "playList": []
  } as PlayStateType)

  // 订阅事件
  async aboutToAppear() {
    this.initUser()
    // 播放器初始化（核心）
    await AvPlayerUtils.init()
    //订阅事件，得到数据   emitter.on 定义
    emitter.on({ eventId: EmitEventType.UPDATE_STATE }, (data) => {
      console.log('666', data.data!.playStateStr)
      this.playState = new PlayStateModel(JSON.parse(data.data!.playStateStr))
      console.log('logData999', '得到通知:::', this.playState)
    })

  }

  //获得我的歌单
  async getMySongData() {
    const response = await geMySongList()
    if (response.success) {
      let data = JSON.stringify(response.data)
      this.mySongList = JSON.parse(data)
      if (this.isLogin) {
        this.isCollectList = await CollectService.isCollectData(this.mySongList.songs)
      }
    } else {
      promptAction.showToast({
        message: response.message,
        duration: 5000
      })
    }
  }

  async getSongCommentData() {
    this.momentList = await CommentService.getSongCommentData()
  }

  //获取用户登录状态
  async initUser() {
    let tempUserInfo = await userService.getUserData()
    if (tempUserInfo.username != undefined) {
      this.isLogin = true
      this.userInfo = tempUserInfo
      PersistentStorage.persistProp("isLogin", this.isLogin)
      PersistentStorage.persistProp("userInfo", this.userInfo)
      console.log(JSON.stringify(tempUserInfo) + '用户信息===============')
    }
  }

  @Builder
  TabBarWidget(item: TabBarModel) {
    Column() {
      Row() {
        Image(item.icon)
          .fillColor(Color.White)
          .width(item.index === this.currentIndex ? '35lpx' : '40lpx')
          .aspectRatio(1)
      }
      .width('40lpx')
      .aspectRatio(1)
      .borderRadius('40lpx')
      .backgroundColor(item.index === this.currentIndex ? '#753061' : Color.Transparent)
      .justifyContent(FlexAlign.Center)

      Text(item.title)
        .fontSize('22lpx')
        .fontWeight(FontWeight.Bold)
        .margin({ top: '10lpx' })
        .fontColor(item.index === this.currentIndex ? '#753061' : '#ffffff')
    }
    .width('100%')
    .justifyContent(FlexAlign.Center)
    .margin({ top: '18lpx' })
  }

  build() {
    Stack({ alignContent: Alignment.Bottom }) {
      Tabs({ barPosition: BarPosition.End, index: this.currentIndex }) {
        ForEach(tabsData, (item: TabBarModel) => {
          TabContent() {
            if (item.index == 0) {
              Home()
            } else if (item.index == 1) {
              LikeSongList({
                mySongList: this.mySongList,
                isCollectList: this.isCollectList,
                func: () => {
                  this.getMySongData()
                }
              })
            } else if (item.index == 2) {
              Comment({
                momentList: this.momentList,
                func: () => {
                  this.getSongCommentData()
                }
              })
            } else if (item.index == 3) {
              Mine()
            }

          }
          .tabBar(this.TabBarWidget(item))
          .backgroundColor('#eeeeee')
        }, (item: TabBarModel) => item.name)
      }
      .barHeight('110lpx')
      .onChange(index => {
        this.currentIndex = index
        if (index == 1) { //点击tab切换进入到
          // 更新我的歌单
          this.getMySongData()
        } else if (index == 2) {
          this.getSongCommentData()
        }
      })

      // 背景播放
      if (this.currentIndex !== 3) {
        // 父向子传值
        PlayerNav({ playState: $playState })
          .translate({ y: '-110lpx' })
          .onClick(() => {
            if (!this.isLogin) {
              promptAction.showToast({
                message: '请登录',
                duration: 5000
              })
            } else {
              router.pushUrl({
                url: 'pages/Play'
              })
            }
          })
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#4781c3')
  }
}
