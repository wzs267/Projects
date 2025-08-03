import searchService from '../services/SearchService'
import router from '@ohos.router';
import { SongListModel,SongListDetailModel } from '../model/AppModel'
import inputMethod from '@ohos.inputMethod'
@Component
export struct SearchNav {
  @Link keywords: string
  @Link songList: SongListModel[] |SongListDetailModel[]
  @Link page: number
  @Link hasMore: boolean
  @Link selectedSubHeaderIndex: number
  getSongListData:()=>void = ()=>{}  //接收父组件的方法

  build() {
    Column() {
      //导航
      Row() {
        Row() {
          Image($r("app.media.back")).height('44lpx').padding({ left:'12lpx' }).fillColor(Color.Gray)
        }.width('40lpx')
        .height('100%')
        .onClick(() => {
          router.back()
        })
        Row() {
          Search({ value: this.keywords,
            placeholder: '查询 歌曲 ,歌单' })
            .height('96lpx')
            .backgroundColor('#FCF3EC')
            .placeholderColor(Color.Grey)
            .placeholderFont({ size: "38lpx", weight:FontWeight.Bold })
            .textFont({ size: "38lpx", weight: 400 })
            .margin({
              left: "44lpx",
              right: "44lpx"
            })
            .onChange(async (value) => {
              //保存搜索关键词
              await searchService.setHistoryData(value)
              //执行搜索
              this.keywords = value
              //重置数据
              this.songList = []
              this.page = 1
              this.hasMore = true
              //重新请求数据
              if (this.selectedSubHeaderIndex==0) {
                this.getSongListData()
              }
              //隐藏键盘
              let im=inputMethod.getController()
              im.stopInputSession()
            })
        }.layoutWeight(1)
      }.width("100%").height("140lpx")
      .padding({
        left: "34lpx",
        right: "34lpx"
      }).justifyContent(FlexAlign.SpaceBetween)
    }
    .backgroundColor(Color.White)
  }
}