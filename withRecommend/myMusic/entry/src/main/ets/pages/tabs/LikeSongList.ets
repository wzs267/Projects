import { MySongModel, UserModel, SongModel } from '../model/AppModel'
import router from '@ohos.router';
import CommentService from '../services/CommentService'
import { SongItem } from '../widget/SongItem'
import { AvPlayerUtils } from '../utils/AvPlayerUtils'
import { geMySongList } from '../common/Api'
import promptAction from '@ohos.promptAction'
import CollectService from '../services/CollectService'

@Component
export struct LikeSongList {
  @StorageLink("isLogin") isLogin: boolean = false
  @StorageLink("userInfo") userInfo: UserModel = {}
  @Prop mySongList: MySongModel = new MySongModel({ username: '', avatar: '' }, [])
  @State isShowComment: boolean = false
  @State content: string = ''
  @State songId: number = -1
  @Prop isCollectList: boolean[] = []
  func: () => void = () => {
  }
  onPageShow() {
    this.getMySongData()
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

  @Builder
  myBuilder() {
    Column() {
      Column() {
      }.height('100lpx')

      Row({ space: '20lpx' }) {
        TextInput({ placeholder: '请输入内容' })
          .layoutWeight(1)
          .onChange((value) => {
            this.content = value
          })
        Button('发布')
          .fontSize('40lpx')
          .onClick(() => {
            CommentService.addCommentData({ content: this.content, up: 0, songId: this.songId })
            this.isShowComment = false;
          })
      }
      .margin({ bottom: '20lpx' })
      .justifyContent(FlexAlign.SpaceBetween)
    }
    .width('100%')
  }
  build() {
    if (this.isLogin) {
      Column() {
        // 播放页
        Column() {
          // 歌单明细
          Column() {
            // 简介
            Row({ space: '16lpx' }) {
              Row() {
                Image($r('app.media.ic_favorite'))
                  .width('80%')
                  .opacity(0.8)
                  .fillColor('#ff5186')
              }
              .width('200lpx')
              .aspectRatio(1)
              .borderRadius('16lpx')
              .backgroundColor(Color.White)
              .justifyContent(FlexAlign.Center)

              Column({ space: '24lpx' }) {

                Text('我喜欢的音乐')
                  .fontWeight(FontWeight.Bold)
                  .fontColor(Color.White)
                Text(`普通会员：${this.userInfo.username}`)
                  .fontWeight(FontWeight.Bold)
                  .fontColor(Color.White)
                  .fontSize('24lpx')
                Blank()
              }
              .alignItems(HorizontalAlign.Start)
              .layoutWeight(1)
            }

            // 按钮
            Row() {
              Button() {
                Row({ space: '16lpx' }) {
                  Image($r('app.media.ic_share'))
                    .width(20)
                    .fillColor('#ec5c87')
                  Text('分享')
                    .fontColor(Color.White)
                    .fontSize('24lpx')
                }
                .width('200lpx')
                .height('72lpx')
                .justifyContent(FlexAlign.Center)
                .backgroundColor('#ff353333')
              }
              .clip(true)
              Button() {
                Row({ space: '16lpx' }) {
                  Image($r('app.media.ic_comment'))
                    .width('40lpx')
                    .fillColor('#ec5c87')
                  Text('评论')
                    .fontColor(Color.White)
                    .fontSize('24lpx')
                }
                .width('200lpx')
                .height('72lpx')
                .justifyContent(FlexAlign.Center)
                .backgroundColor('#ff353333')
              }
              .clip(true)

              Button() {
                Row({ space: '16lpx' }) {
                  Image($r('app.media.ic_collect'))
                    .width('40lpx')
                    .fillColor('#ec5c87')
                  Text('收藏')
                    .fontColor(Color.White)
                    .fontSize('24lpx')
                }
                .width('200lpx')
                .height('72lpx')
                .justifyContent(FlexAlign.Center)
                .backgroundColor('#ff353333')
              }
              .clip(true)
            }
            .width('100%')
            .justifyContent(FlexAlign.SpaceBetween)
          }
          .width('100%')
          .height('30%')
          .justifyContent(FlexAlign.SpaceAround)
          .radialGradient({
            center: ['80%', '-10%'],
            radius: '100%',
            colors: [['#ff865d13', 0.0], [Color.Transparent, 1.0]]
          })
          .padding(16)

          // 播放列表
          Column() {
            Row({ space:'16lpx' }) {
              Row({ space: '16lpx' }) {
                Image($r("app.media.ic_play"))
                  .width('32lpx')
                  .fillColor('#ff5186')
                Text(`播放全部 (${this.mySongList.songs.length})`)
                  .fontColor('#333333')
                  .fontSize('28lpx')
              }
            }
            .width('100%')
            .padding('32lpx')
            .border({
              width: { bottom: 1 },
              color: '#12ec5c87'
            })

            Column() {
              List({ space: '30lpx' }) {
                ForEach(this.mySongList.songs, (item: SongModel, index: number) => {
                  ListItem() {
                    SongItem({
                      item: item,
                      index: index,
                      isCollectList: this.isCollectList,
                      func: () => {
                        // this.getMySongData()
                        this.func()
                      },
                      isShowComment: this.isShowComment,
                      songId: this.songId
                    })
                  }
                  .border({
                    width: { bottom: '2lpx' },
                    color: '#12ec5c87'
                  })
                  .onClick(() => {
                    //点击播放
                    console.log('+++++playone')
                    AvPlayerUtils.playSingle(item)
                  })
                })
                ListItem() {
                  Row() {
                    Text('我是有底线的~')
                      .fontColor(Color.Gray)
                  }
                  .width('100%')
                  .justifyContent(FlexAlign.Center)
                  .padding('32lpx')
                }
              }
              .height('100%')
              .width('100%')
            }
            .layoutWeight(1)
            .bindSheet(this.isShowComment, this.myBuilder(), {
              height: '400lpx',
              backgroundColor: '#ffffff',
              maskColor: Color.Gray,
              onDisappear: () => {
                this.isShowComment = false; // 确保任何方式关闭弹窗时状态都被更新
              }
            })
          }
          .layoutWeight(1)
          .width('100%')
          .backgroundColor('#cccccc')
          .borderRadius({ topLeft: '24lpx', topRight: '24lpx' })
        }
        .layoutWeight(1)
      }
      .width('100%')
      .height('100%')
      .backgroundColor('#eeeeee')
    } else {
      Button('请求登录')
        .onClick(() => {
          router.pushUrl({
            url: "pages/Login"
          })
        })
        .backgroundColor('#4781c3')
    }
  }
}

