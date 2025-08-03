import { PlayStateType } from '../model/PlayStateModel'
import {ConfigUtil as Config} from '../common/ConfigUtil'
@Component
export default struct PlayerNav {
  @StorageLink("isLogin") isLogin: boolean = false
  @Link playState: PlayStateType
  build() {
    Row() {
      Row({space:'20lpx'}) {
        Image(Config.SERVER_URL+this.playState.pic)
          .width('80lpx')
          .aspectRatio(1)
        Column() {
          Text(this.playState.name)
            .fontColor('#fff')
        }.alignItems(HorizontalAlign.Start)
      }.height('100%')
      .alignItems(VerticalAlign.Center)
      Row({space:'32lpx'}) {
        Image($r("app.media.ic_prev"))
          .width('40lpx')
          .aspectRatio(1)
          .fillColor('#4cafc6')
        Image( this.playState.isPlay?$r('app.media.ic_paused'): $r('app.media.ic_play'))
          .fillColor('#4cafc6')
          .width('40lpx')
        Image($r('app.media.ic_next'))
          .width('40lpx')
          .aspectRatio(1)
          .fillColor('#4cafc6')
      }.height('100%')
    }
    .width('100%')
    .height('120lpx')
    .backgroundColor('#ff1c1c21')
    .justifyContent(FlexAlign.SpaceBetween)
    .padding({ left: '40lpx', right: '40lpx' })
  }
}
