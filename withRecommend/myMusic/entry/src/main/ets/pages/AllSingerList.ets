import { SingerModel} from './model/AppModel'
import { ConfigUtil as Config } from './common/ConfigUtil'
import { ParamModel } from './model/paramModel'
import {getAllSinger} from './common/Api'
import { NavBar } from './widget/NavBar'
import router from '@ohos.router';
@Entry
@Component
struct AllSingerList {
  @State singerList: SingerModel[] = []
  // 获取歌手
  async getSingerData () {
    let params:ParamModel ={
      page:1
    }
    const response = await getAllSinger(params)
    if (response.success) {
      let data = JSON.stringify(response.data)
      this.singerList = JSON.parse(data)
      console.log('提交歌手', this.singerList)
    }
  }
  aboutToAppear() {
    this.getSingerData()
  }
  build() {
    Column() {
      NavBar({title:'歌手列表'})
      Row() {
        Text('歌手列表').fontColor('#333333')
      }
      .padding('48lpx')
      .width('100%')
      Row(){
        List({space:'24lpx'}) {
          ForEach(this.singerList,(item:SingerModel)=>{
            ListItem() {
              Row({space:'24lpx'}) {
                Row() {
                  Image(Config.SERVER_URL+item.pic)
                    .borderRadius('16lpx')
                    .width('120lpx')
                }
                Column({space:'24lpx'}) {
                  Text(item.name).fontColor('#333333')
                  Row({space:'8lpx'}){
                    Text('VIP') .fontSize('20lpx')
                      .padding({
                        left:'12lpx',
                        right:'12lpx',
                        top:'4lpx',
                        bottom:'4lpx'
                      })
                      .border({
                        width:1,
                        color:'#c0a267'
                      }).borderRadius('20lpx').fontColor('#c0a267')
                    Text(item.introduction)
                      .fontColor('#aaa9af')
                      .fontSize('24lpx')
                      .maxLines(2) // 最多显示2行
                      .textOverflow({ overflow: TextOverflow.Ellipsis }) // 超出部分显示省略号

                  }
                }
                .layoutWeight(1)
                .alignItems(HorizontalAlign.Start)
                Row() {
                  Image($r('app.media.ic_more'))
                    .fillColor('#aaa9af')
                    .width('48lpx')
                }
              }
              .onClick(() => {
                router.pushUrl({
                  url: 'pages/SingerDetail',
                  params: {
                    cid: item.id
                  }
                })
              })
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
        }.width('100%').height('100%')
        .padding({
          left:'48lpx',
          right:'48lppx'
        })
      }
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
  }
}
