import router from '@ohos.router';
import {ParamModel} from './model/paramModel'
import {SongListDetailModel,SongListModel,SongModel,UserModel} from './model/AppModel'
import {getListSongDetail} from './common/Api'
import {ConfigUtil as Config} from './common/ConfigUtil'
import promptAction from '@ohos.promptAction'
import {NavBar} from './widget/NavBar'
import {SongItem} from './widget/SongItem'
import CollectService from './services/CollectService'
import CommentService from './services/CommentService'

@Entry
@Component

struct SingerDetail {
  @StorageLink("isLogin") isLogin: boolean = false
  @StorageLink("userInfo") userInfo: UserModel = {}
  @State cid: number = 0
  @State SongListDetail:SongListDetailModel ={songList:new SongListModel(0,"","","","","",""),songs:[]}
  @State isShowComment: boolean = false
  @State content: string = ''
  @State songId :number= -1
  @State isCollectList: boolean[] = []
  aboutToAppear() {
    const params:ParamModel=  (router.getParams() as ParamModel)
    this.cid = params.cid?params.cid:0 // 处理undefined
    this.getSongDetailData()
  }
  async getSongDetailData(){
    const params:ParamModel = {
      id:this.cid
    }
    const res = await getListSongDetail(params)
    if (res.success) {
      this.SongListDetail = JSON.parse(JSON.stringify(res.data))
      if (this.isLogin) {
        this.isCollectList= await CollectService.isCollectData(this.SongListDetail.songs)
      }
    }else{
      promptAction.showToast({
        message: res.message,
        duration: 5000
      })
    }
  }
  @Builder
  myBuilder() {
    Column() {
      Column(){}.height('100lpx')
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
    Column() {
      NavBar({title:'歌单详情'})
      // 播放页
      Column() {
        // 歌单明细
        Column() {
          // 简介
          Row({ space:'16lpx' }) {
            Row() {
              Image(Config.SERVER_URL+this.SongListDetail.songList.pic)
                .width('80%')
                .opacity(0.8)
                .fillColor('#ff5186')
            }
            .width('200lpx')
            .aspectRatio(1)
            .borderRadius(8)
            .backgroundColor(Color.White)
            .justifyContent(FlexAlign.Center)

            Column({ space:'24lpx' }) {
              Text(`${this.SongListDetail.songList.title}`)
                .fontWeight(FontWeight.Bold)
                .fontColor(Color.White)
              Blank()

            }
            .alignItems(HorizontalAlign.Start)
            .layoutWeight(1)
          }
          // 按钮
          Row() {
            Button() {
              Row({ space:'16lpx' }) {
                Image($r('app.media.ic_share'))
                  .width('40lpx')
                  .fillColor('#aa5c87')
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
              Row({ space: '16lpx'}) {
                Image($r('app.media.ic_comment'))
                  .width('40lpx')
                  .fillColor('#aa5c87')
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
              Row({ space:'16lpx' }) {
                Image($r('app.media.ic_collect'))
                  .width('40lpx')
                  .fillColor('#aa5c87')
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
        .padding('32lpx')
        // 播放列表
        Column() {
          Row({ space: '16lpx' }) {
            Row({space:'16lpx'}){
              Image($r("app.media.ic_play"))
                .width('32lpx')
                .fillColor('#ff5186')
              Text(`播放全部 (${this.SongListDetail.songs.length})`)
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
            List({space:'20lpx'}) {
              ForEach(this.SongListDetail.songs, (item: SongModel, index: number) => {
                ListItem() {
                  SongItem({
                    item: item,
                    index: index,
                    isCollectList: this.isCollectList,
                    func: ()=>{
                      this.getSongDetailData()
                    },
                    isShowComment: this.isShowComment,
                    songId: this.songId
                  })
                }
                .border({
                  width: { bottom: '2lpx' },
                  color: '#12ec5c87'
                })
                .onClick(()=>{
                  //点击播放
                  console.log('+++++playone')
                  //AvPlayerUtils.playSingle(item)
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
            .width('100%')
            .height('100%')
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
  }
}

