import router from '@ohos.router';
import {ParamModel} from './model/paramModel'
import {SongDetailModel} from './model/AppModel'
import {getSongDetail} from './common/Api'
import {ConfigUtil as Config} from './common/ConfigUtil'
import promptAction from '@ohos.promptAction'
import {NavBar} from './widget/NavBar'
@Entry
@Component
struct SingerDetail {
  @State cid: number = 0
  @State songDetail:SongDetailModel[] =[]
  aboutToAppear() {
    const params:ParamModel=  (router.getParams() as ParamModel)
    this.cid = params.cid?params.cid:0 // 处理undefined
    this.getSingerDetailData()
  }
  async getSingerDetailData(){
    const params:ParamModel = {
      id:this.cid
    }
    const res = await getSongDetail(params)
    if (res.success) {
      this.songDetail = JSON.parse(JSON.stringify(res.data))
      console.log('getSongDetail'+ JSON.stringify(this.songDetail))
    }else{
      promptAction.showToast({
        message: res.message,
        duration: 5000
      })
    }
  }
  build() {
    Column() {
      NavBar({title:'歌曲详情'})
      // 播放页
      Column() {
        // 歌单明细
        Column() {
          // 简介
          Row({ space:'16lpx' }) {
            Row() {
              if (this.songDetail.length>0){
                Image(Config.SERVER_URL+this.songDetail[0].pic)
                  .width('80%')
                  .opacity(0.8)
                  .fillColor('#ff5186')
              }

            }
            .width('200lpx')
            .aspectRatio(1)
            .borderRadius('16lpx')
            .backgroundColor(Color.White)
            .justifyContent(FlexAlign.Center)

            Column({ space: '24lpx' }) {
              if (this.songDetail.length>0) {
                Text(`${this.songDetail[0].name}`)
                  .fontWeight(FontWeight.Bold)
                  .fontColor(Color.White)
              }
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
              Row({ space: '16lpx' }) {
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
          Row({ space:'16lpx' }) {
            Row({space:'16lpx'}){
              Image($r("app.media.ic_play"))
                .width('32lpx')
                .fillColor('#ff5186')
              Text(`播放`)
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
            List() {
              ForEach(this.songDetail, (item: SongDetailModel, index: number) => {
                ListItem() {
                  Row() {
                    // 前三首
                    Row() {
                      if (index === 0) {
                        Text((index + 1).toString())
                          .fontWeight(FontWeight.Bold)
                          .fontColor('#ffffe426')
                      } else if (index === 1) {
                        Text((index + 1).toString())
                          .fontColor('#ff6600')
                          .fontWeight(FontWeight.Bold)
                      } else if (index === 2) {
                        Text((index + 1).toString())
                          .fontWeight(FontWeight.Bold)
                          .fontColor('#ff6600')
                      } else {
                        Text((index + 1).toString())
                          .fontColor('#ffa49a9a')
                      }
                    }
                    .width('100lpx')
                    .aspectRatio(1)
                    .justifyContent(FlexAlign.Center)
                    // 列表
                    Row({ space: '20lpx' }) {
                      Image(Config.SERVER_URL+item.pic)
                        .width('64lpx')
                        .aspectRatio(1)
                        .borderRadius('8lpx')
                      Column() {
                        Text(item.name)
                          .fontSize('28lpx')
                          .fontColor(Color.Black)
                        Text(item.name)
                          .fontSize('24lpx')
                          .fontColor(Color.Gray)
                      }
                      .layoutWeight(1)
                      .alignItems(HorizontalAlign.Start)
                      .justifyContent(FlexAlign.Center)
                    }
                    .layoutWeight(1)

                    Image($r('app.media.ic_more'))
                      .width('48lpx')
                      .height('48lpx')
                      .margin({ right: '32lpx' })
                      .fillColor(Color.Gray)
                  }
                  .alignItems(VerticalAlign.Center)
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

