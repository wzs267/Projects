import router from '@ohos.router'; // 路由跳转
import { UserModel } from '../model/AppModel'
import userService from '../services/UserService'
import { ConfigUtil as Config } from '../common/ConfigUtil'
import promptAction from '@ohos.promptAction'
import { AvPlayerUtils } from '../utils/AvPlayerUtils';
import { PlayStateModel, PlayStateType } from '../model/PlayStateModel'

interface ListItem {
  icon: Resource; // 图标资源
  name: string; // 标题
  route: string; // 描述
}

@Component
export struct Mine {
  @StorageLink("isLogin") isLogin: boolean = false
  @StorageLink("userInfo") userInfo: UserModel = {}
  @State
  playState:PlayStateType = new PlayStateModel({"pic":"/img/songPic/tubiao.jpg","name":"音乐网","url":"","playIndex":0,"time":0,"duration":0,"isPlay":false,"playMode":"auto","playList":[]} as PlayStateType)
  //退出登录
  async loginOut() {
    //清除用户数据
    userService.cleanUserData()
    this.isLogin = false
    this.userInfo = {}
    AvPlayerUtils.player.reset()
    AvPlayerUtils.playList=[]
    AvPlayerUtils.isPlay = false
    AvPlayerUtils.duration = 0
    AvPlayerUtils.time = 0
    AvPlayerUtils.updateState(this.playState)
  }
  // 功能选项数据
  @State list: ListItem[] = [
    { icon: $r('app.media.ic_favorite'), name: '我的歌单', route: 'pages/Index' },
    { icon: $r('app.media.ic_music'), name: '我的浏览记录', route: '' },
    { icon: $r('app.media.setting'), name: '设置', route: 'pages/UserEdit' }
  ];

  build() {
    Column() {
      // 1. 顶部用户信息区
      this.buildUserInfoArea()

      // 2. 功能选项列表
      List() {
        ForEach(this.list, (item: ListItem) => {
          ListItem() {
            this.buildFunctionItem(item)
          }
          .onClick(() => {
            if (!this.isLogin) {
              promptAction.showToast({
                message: '请登录',
                duration: 5000
              })
              return;
            }
            // 跳转至对应页面
            if (item.route === 'pages/Index') {
              router.pushUrl({
                url: item.route,
                params: {
                  tabIndex: 1
                }
              });
            } else {
              router.pushUrl({ url: item.route })
            }

          })
        })
      }
      .padding('32lpx')
      .margin({ top: '40lpx' })
      .backgroundColor('#F5F5F5')
      .borderRadius('24lpx')
      .height('400lpx')
      .width('100%')

      // 3. 底部信息
      this.buildFooter()
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#EEEEEE')
  }

  // 构建用户信息区
  @Builder
  buildUserInfoArea() {
    Column() {
      // 背景图
      Image($r('app.media.found'))
        .width('100%')
        .height('400lpx')
        .objectFit(ImageFit.Cover)
        .zIndex(1)
      //用户头像 和用户名
      Row() {
        if (this.isLogin) {
          Image(`${Config.SERVER_URL}${this.userInfo.avatar}`).width("148lpx").height("148lpx").borderRadius("74lpx")
        } else {
          Image($r("app.media.user")).width("148lpx").height("148lpx").borderRadius("74lpx").onClick(() => {
            router.pushUrl({
              url: "pages/Login"
            })
          })
        }
        Column() {
          if (this.isLogin) {
            Column() {
              Text(`${this.userInfo.username}`)
                .fontSize("38lpx")
              Text(`${this.userInfo.introduction}`)
                .fontSize("26lpx")
                .fontColor('#666666')
            }.alignItems(HorizontalAlign.Start)
            .height("148lpx")
            .justifyContent(FlexAlign.SpaceEvenly)
          } else {
            Text("登录/注册").fontSize("33lpx").fontColor('#666666').width("100%").onClick(() => {
              router.pushUrl({
                url: "pages/pass/CodeLoginStepOne"
              })
            })
          }
        }
        .width("600lpx")
        .alignItems(HorizontalAlign.Start)
        .height("148lpx")
        .justifyContent(FlexAlign.SpaceAround)
        .margin({
          left: "40lpx"
        })

      }.height("188lpx").width("100%").position({ x: '20%', y: 70 }) // 调整位置覆盖背景
      .zIndex(2)

    }
    .width('100%')
  }

  // 构建功能选项Item
  @Builder
  buildFunctionItem(item: ListItem) {
    Row() {
      Image(item.icon)
        .width('48lpx')
        .height('48lpx')
        .fillColor(Color.Grey)
      Text(item.name)
        .fontSize('32lpx')
        .margin({ left: '32lpx' })
        .fontColor('#333333')
      Blank() // 推到右侧
      Image($r('app.media.right_arrow'))
        .width('32lpx')
        .height('32lpx')
        .opacity(0.6)
    }
    .padding('24lpx')
    .width('100%')
  }

  // 构建底部信息
  @Builder
  buildFooter() {
    Column() {
      Button('退出登录')
        .fontSize('28lpx')
        .backgroundColor('#4781c3')
        .margin({ top: '40lpx' })
        .onClick(() => {
          if (!this.isLogin) {
            promptAction.showToast({
              message: '请登录',
              duration: 5000
            })
            return;
          }
          this.loginOut()
        })
    }
  }
}