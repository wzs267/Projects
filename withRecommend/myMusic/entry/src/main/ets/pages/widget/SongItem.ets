import {SongModel,UserModel} from '../model/AppModel'
import {ConfigUtil as Config} from '../common/ConfigUtil'
import CollectService from '../services/CollectService'
import promptAction from '@ohos.promptAction'
@Component
export struct SongItem {
  @StorageLink("isLogin") isLogin: boolean = false
  @StorageLink("userInfo") userInfo: UserModel = {}
  @ObjectLink item:  SongModel
  @Prop index: number
  @Link isCollectList: boolean[]
  @Link isShowComment :boolean
  @Link songId: number
  func:()=>void = ()=>{}
  build() {
    Row() {
      // 前三首
      Row() {
        if (this.index === 0) {
          Text((this.index + 1).toString())
            .fontWeight(FontWeight.Bold)
            .fontColor('#ffffe426')
        } else if (this.index === 1) {
          Text((this.index + 1).toString())
            .fontColor('#ffffcc00')
            .fontWeight(FontWeight.Bold)
        } else if (this.index === 2) {
          Text((this.index + 1).toString())
            .fontWeight(FontWeight.Bold)
            .fontColor('#ffff9900')
        } else {
          Text((this.index + 1).toString())
            .fontColor('#ffa49a9a')
        }
      }
      .width('60lpx')
      .aspectRatio(1)
      .justifyContent(FlexAlign.Center)

      // 列表
      Row({ space: '20lpx' }) {
        Image(Config.SERVER_URL+this.item.pic)
          .width('36lpx')
          .aspectRatio(1)
          .borderRadius('8lpx')

        Column() {
          Text(this.item.name)
            .fontSize('24lpx')
            .fontColor(Color.Black)
        }
        .layoutWeight(1)
        .alignItems(HorizontalAlign.Start)
        .justifyContent(FlexAlign.Center)
        Image($r('app.media.ic_like'))
          .width('48lpx')
          .height('48lpx')
          .margin({ right: '32lpx' })
          .fillColor(this.isCollectList[this.index]?Color.Red:Color.Gray)
          .onClick(async()=>{
            if (!this.isLogin){
              promptAction.showToast({
                message: '请登录',
                duration: 5000
              })
            }else{
              const flag= await CollectService.setCollectData(this.item.id,this.isCollectList[this.index]?'cancel':'add')
              flag && (this.isCollectList[this.index]=!this.isCollectList[this.index])
              this.func()
            }

          })
        Image($r('app.media.ic_comment_o'))
          .width('48lpx')
          .height('48lpx')
          .onClick(()=>{
            if (!this.isLogin){
              promptAction.showToast({
                message: '请登录',
                duration: 5000
              })
            }else{
              this.isShowComment = !this.isShowComment
              this.songId = this.item.id
            }

          })
        Image($r('app.media.ic_single_play'))
          .width('48lpx')
          .height('48lpx')
          .margin({ right: '32lpx' })
          .fillColor(Color.Gray)
      }
      .layoutWeight(1)
    }
    .alignItems(VerticalAlign.Center)
  }
}