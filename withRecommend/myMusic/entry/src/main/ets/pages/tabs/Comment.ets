import { ConfigUtil as Config } from '../common/ConfigUtil'
import { SongCommentModel } from '../model/AppModel'
import { getReplyComment, } from '../common/Api'
import promptAction from '@ohos.promptAction'
import CommentService from '../services/CommentService'
import { AvPlayerUtils } from '../utils/AvPlayerUtils'
@Component
export struct Comment {
  @State isShow: boolean = false
  @Prop momentList: SongCommentModel[] = []
  @State getReplyCommentList: SongCommentModel[] = []
  @State currentId: number = -1
  @State replyComment: string = ''
  @State commentId: number = -1
  @State songId: number = -1
  scroller = new Scroller()
  func:()=>void = ()=>{}

  // 通过评论ID 获取回复评论列表getReplyComment
  async getReplyCommentData(id: number) {
    const response = await getReplyComment(id)
    if (response.success) {
      let data = JSON.stringify(response.data)
      console.log("获取歌曲评论列表：" + data)
      this.getReplyCommentList = JSON.parse(data)
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
      Column(){
        Scroll(this.scroller) {
          Column() {
            List({ space: '24lpx' }) {
              ForEach(this.getReplyCommentList, (item: SongCommentModel) => {
                ListItem() {
                  Column() {
                    Row({ space: '24lpx' }) {
                      Row() {
                        Image(Config.SERVER_URL + item.consumer.avatar)
                          .borderRadius('80lpx')
                          .width('80lpx')
                      }
                      Column({ space: '24lpx' }) {
                        // 作者
                        Text(item.consumer.username)
                          .fontColor('#333333')
                        // 内容
                        Text(item.content)
                          .fontColor('#333333')
                          .fontSize('24lpx')
                      }
                    }
                  }
                }
              }, (item: SongCommentModel) => item.id.toString())
            }
            .width('100%')
            .height('100%')
          }
        }
      }.layoutWeight(1)
       .margin({ top: '100lpx',bottom:'60lpx' })
      Row({ space: '20lpx'}) {
        TextInput({ placeholder: '请输入内容' })
          .layoutWeight(1)
          .onChange((value) => {
            this.replyComment = value
          })
        Button('发布')
          .fontSize('40lpx')
          .onClick(() => {
            CommentService.addCommentData({ content: this.replyComment, up: this.commentId, songId: this.songId })
            //this.getSongCommentData()
            this.func()
            this.getReplyCommentData(this.commentId)
            this.isShow = false;
          })
      }
      .margin({ bottom: '40lpx' })
      .justifyContent(FlexAlign.SpaceBetween)

    }
    .width('100%')
  }

  build() {
    Stack({ alignContent: Alignment.BottomEnd }) {
      Column() {
        Row() {
          Text('互动广场')
            .fontColor('#333333').fontSize('44lpx').fontWeight(FontWeight.Bold)
        }
        .padding('48lpx')
        .width('100%')

        Row() {
          List({ space: '24lpx' }) {
            ForEach(this.momentList, (item: SongCommentModel) => {
              ListItem() {
                Row({ space: '24lpx' }) {
                  Row() {
                    Image(Config.SERVER_URL + item.consumer.avatar)
                      .borderRadius('80lpx')
                      .width('80lpx')
                  }
                  Column({ space: '24lpx' }) {
                    // 作者
                    Text(item.consumer.username)
                      .fontColor('#333333')
                    // 内容
                    Text(item.content)
                      .fontColor('#333333')
                      .fontSize('24lpx')
                    // 音乐卡片
                    Row() {
                      Image(Config.SERVER_URL + item.song.pic)
                        .width('120lpx')
                        .aspectRatio(1)
                      Column({ space: '20lpx' }) {
                        Text(item.song.name)
                          .fontColor('#f4f4f6')
                          .fontSize('24lpx')
                        Text(item.singer)
                          .fontColor('#a8a8ad')
                          .fontSize('20lpx')
                      }
                      .layoutWeight(1)
                      .height('120lpx')
                      .alignItems(HorizontalAlign.Start)
                      .justifyContent(FlexAlign.SpaceBetween)
                      .padding('24lpx')

                      Image($r("app.media.ic_single_play"))
                        .width('40lpx')
                        .fillColor('#ececec')
                        .margin({ right:'24lpx' })
                    }
                    .backgroundColor('#666666')
                    .width('100%')
                    .borderRadius('16lpx')
                    .clip(true)
                    .onClick(() => {
                      //点击播放
                      console.log('+++++playone')
                      AvPlayerUtils.playSingle(item.song)
                    })

                    Row({ space: '40lpx' }) {
                      Row() {
                        Image($r('app.media.ic_share'))
                          .width('40lpx')
                          .fillColor('#c9c8cd')
                        Text('分享')
                          .fontColor('#333333')
                          .fontSize('24lpx')
                          .margin({ left: '8lpx' })
                      }
                      Row() {
                        Image($r('app.media.ic_comment_o'))
                          .width('40lpx')
                          .fillColor('#333333')
                          .onClick(() => {
                            this.isShow = true
                            this.songId = item.song_id
                            this.getReplyCommentData(item.id)
                            this.commentId = item.id
                            console.log('item.isShow', item.isShow)

                          })
                        Text(item.commentCount.toString())
                          .fontColor('#333333')
                          .fontSize('24lpx')
                          .margin({ left: '8lpx' })

                      }
                      Row() {
                        Image($r('app.media.ic_like'))
                          .width('40lpx')
                          .fillColor('#333333')
                        Text(item.likeCount.toString())
                          .fontColor('#333333')
                          .fontSize('24lpx')
                          .margin({ left:'8lpx' })
                      }
                    }
                  }
                  .layoutWeight(1)
                  .alignItems(HorizontalAlign.Start)
                }
                .alignItems(VerticalAlign.Top)
                .padding({
                  top: '48lpx',
                  bottom: '48lpx'
                })
                .borderWidth({
                  bottom: 1
                })
                .borderColor('#292931')
              }
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
          .width('100%')
          .height('100%')
          .padding({ right: '64lpx', left: '32lpx' })

        }
        .layoutWeight(1)
        .bindSheet(this.isShow, this.myBuilder(), {
          height:this.getReplyCommentList.length>0?'60%':'300lpx',
          backgroundColor: '#ffffff',
          maskColor: Color.Gray,
          onDisappear: () => {
            this.isShow = false; // 确保任何方式关闭弹窗时状态都被更新
          }
        })
      }
      .width('100%')
      .height('100%')
      .backgroundColor('#cccccc')
    }
    .width('100%')
    .height('100%')
  }
}

