import {SwiperModel,SongListModel,SingerModel,RecommendSongModel,SongModel} from '../model/AppModel'
import {getSwiperList,getRecommendSongList,getRecommendSinger,getRecommendSongs} from '../common/Api'
import {ConfigUtil as Config} from '../common/ConfigUtil'
import router from '@ohos.router';
import { PlayStateModel } from '../model/PlayStateModel'
import emitter from '@ohos.events.emitter'
import { EmitEventType } from '../model/paramModel'

@Component
export struct Home{
  @State swiperList:SwiperModel[] = []
  @State songList:SongListModel[] = []
  @State recommendSongs:RecommendSongModel[] = [] // 新增：AI推荐单曲
  @State singerList:SingerModel[] = []
  @StorageLink("userInfo") userInfo: Record<string, ESObject> = {}  // 获取用户信息用于个性化推荐
  // 获取轮播图接口
  async getSwiperData(){  /// async...await
    const res = await getSwiperList()
    this.swiperList = JSON.parse(JSON.stringify(res.data))
    console.log('swiperList'+ JSON.stringify(this.swiperList))
  }
  // 获取推荐歌单接口 -> 改为获取AI推荐单曲
  async recommendSongListData(){
    try {
      // 优先使用AI推荐
      const userId = (this.userInfo?.user?.id as number) || 3; // 默认用户ID为3
      const res = await getRecommendSongs(userId, 6);
      // 检查响应是否包含code属性（后端返回格式）或success属性（ResponseModel格式）
      const resRecord = res as Record<string, ESObject>;
      const isSuccess = resRecord.code === 200 || res.success;
      const responseData: ESObject = resRecord.data || res.data;
      
      if (isSuccess && responseData) {
        this.recommendSongs = responseData as RecommendSongModel[];
        console.log('AI推荐单曲:', JSON.stringify(this.recommendSongs));
      } else {
        // 降级到原有歌单推荐
        const fallbackRes = await getRecommendSongList();
        this.songList = JSON.parse(JSON.stringify(fallbackRes.data));
      }
    } catch (error) {
      console.error('推荐获取失败:', error);
      // 降级到原有歌单推荐
      const fallbackRes = await getRecommendSongList();
      this.songList = JSON.parse(JSON.stringify(fallbackRes.data));
    }
  }
  // 获取推荐歌手接口
  async recommendSingerData(){
    const res = await getRecommendSinger()
    this.singerList = JSON.parse(JSON.stringify(res.data))

  }
  // 生命周期
  aboutToAppear(): void {
    this.getSwiperData()
    this.recommendSongListData()
    this.recommendSingerData()
  }
  @Builder
  getSwiperListWidget(){
    Swiper() {
      ForEach(this.swiperList, (item: SwiperModel) => {
        Image(`${Config.SERVER_URL}${item.imgurl}`)
          .width('100%')
          .height('400lpx')
          .objectFit(ImageFit.Cover)
      }, (item: SwiperModel) => item.id.toString())
    }
    .loop(true)  //无限循环
    .autoPlay(true) // 自动播放
    .interval(2000) // 轮播间隔时间
    //.displayArrow(true)
    .indicator(  // 指示器
      Indicator.dot()
        .left(0)
        .itemWidth('30lpx')
        .itemHeight('30lpx')
        .selectedItemWidth('60lpx')
        .selectedItemHeight('30lpx')
        .color('#999999')
        .selectedColor('#ff6600')
    )
    .height('460lpx')
  }
  @Builder
  getSongListWidget(){
    //   AI推荐单曲
    Column() {
      this.TitleBar('AI推荐单曲')
      // 推荐单曲列表
      if (this.recommendSongs.length > 0) {
        // 使用AI推荐结果
        Column({ space: '20lpx' }) {
          ForEach(this.recommendSongs, (item: RecommendSongModel) => {
            Row() {
              // 歌曲封面
              Image(Config.SERVER_URL + item.pic)
                .width('120lpx')
                .height('120lpx')
                .borderRadius('12lpx')
                .objectFit(ImageFit.Cover)
              
              // 歌曲信息
              Column() {
                Row() {
                  Text(item.name)
                    .fontSize('32lpx')
                    .fontWeight(FontWeight.Medium)
                    .fontColor('#FF000000')
                    .maxLines(1)
                    .textOverflow({ overflow: TextOverflow.Ellipsis })
                  
                  // 推荐标签
                  Text('AI推荐')
                    .fontSize('20lpx')
                    .fontColor('#FF007DFF')
                    .backgroundColor('#1A007DFF')
                    .padding({ left: '8lpx', right: '8lpx', top: '4lpx', bottom: '4lpx' })
                    .borderRadius('8lpx')
                }
                .width('100%')
                .justifyContent(FlexAlign.SpaceBetween)
                .alignItems(VerticalAlign.Center)
                
                if (item.singer?.name) {
                  Text(item.singer.name)
                    .fontSize('24lpx')
                    .fontColor('#FF666666')
                    .margin({ top: '8lpx' })
                }
                
                if (item.reason) {
                  Text(item.reason)
                    .fontSize('20lpx')
                    .fontColor('#FF999999')
                    .margin({ top: '8lpx' })
                }
              }
              .layoutWeight(1)
              .alignItems(HorizontalAlign.Start)
              .margin({ left: '24lpx' })
              
              // 播放按钮
              Text("▶")
                .width('48lpx')
                .height('48lpx')
                .fontSize('24lpx')
                .fontColor('#FF007DFF')
                .textAlign(TextAlign.Center)
                .borderRadius('24lpx')
                .backgroundColor('#F0F0F0')
            }
            .width('100%')
            .padding('16lpx')
            .backgroundColor('#FFFFFFFF')
            .borderRadius('16lpx')
            .shadow({
              radius: 8,
              color: '#1F000000',
              offsetX: 0,
              offsetY: 2
            })
            .onClick(() => {
              // 播放推荐歌曲
              this.playRecommendSong(item);
            })
          })
        }
        .padding({
          left: '16lpx',
          right: '16lpx'
        })
      } else {
        // 降级显示原有歌单
        Row() {
          Flex({
            wrap:FlexWrap.Wrap
          }){
            ForEach(this.songList, (item: SongListModel) => {
              Column() {
                Row() {
                  Image(Config.SERVER_URL +item.pic)
                    .width('100%')
                    .aspectRatio(1)
                    .borderRadius('16lpx')
                  Text(item.style)
                    .fontColor(Color.White)
                    .fontSize('24lpx')
                    .fontWeight(FontWeight.Bold)
                    .position({
                      x: '12lpx',
                      y: '12lpx'
                    })
                }
                Text(item.title)
                  .textAlign(TextAlign.Start)
                  .fontSize('24lpx')
                  .fontColor(Color.Black)
                  .maxLines(2)
                  .textOverflow({
                    overflow: TextOverflow.Ellipsis
                  })
                  .margin({
                    top: '16lpx'
                  })
              }
              .justifyContent(FlexAlign.SpaceBetween)
              .width('33%')
              .onClick(()=>{
                router.pushUrl({
                  url: 'pages/SongListDetail',
                  params: {
                    cid: item.id
                  }
                })
              })
            })
          }
        }
        .padding({
          left: '16lpx',
          right: '16lpx'
        })
      }
    }
  }
  
  // 播放推荐歌曲的方法
  playRecommendSong(song: RecommendSongModel) {
    const playList = [song]; // 将推荐歌曲作为播放列表
    const playState = new PlayStateModel({
      pic: song.pic,
      name: song.name,
      url: song.url,
      playIndex: 0,
      time: 0,
      duration: 0,
      isPlay: true,
      playMode: "auto",
      playList: playList
    });
    
    // 发送播放事件
    emitter.emit({ eventId: EmitEventType.UPDATE_STATE }, {
      data: { playStateStr: JSON.stringify(playState) }
    });
    
    console.log('开始播放AI推荐歌曲:', song.name);
  }
  @Builder
  getSingerWidget(){
    // 推荐歌手
    Column() {
      this.TitleBar('推荐歌手')
      // 歌单项
      Row() {
        Scroll() {
          Row({ space: '16lpx' }) {
            ForEach(this.singerList, (item: SingerModel) => {
              Column() {
                Row() {
                  Image(Config.SERVER_URL+item.pic)
                    .width('100%')
                    .borderRadius('16lpx')
                  Text(item.name)
                    .fontColor(Color.White)
                    .fontSize('24lpx')
                    .fontWeight(FontWeight.Bold)
                    .position({
                      x: '12lpx',
                      y: '12lpx'
                    })
                }
                Text(item.name)
                  .textAlign(TextAlign.Start)
                  .fontSize('24lpx')
                  .fontColor('#FFE9DFDF')
                  .maxLines(2)
                  .textOverflow({
                    overflow: TextOverflow.Ellipsis
                  })
                  .margin({
                    top: '16lpx'
                  })
              }
              .justifyContent(FlexAlign.SpaceBetween)
              .width('30%')
              .onClick(()=>{
                router.pushUrl({
                  url: 'pages/SingerDetail',
                  params: {
                    cid: item.id
                  }
                })
              })
            })
          }
        }
        .width('100%')
        .scrollable(ScrollDirection.Horizontal).scrollBar(BarState.Off)
      }
      .padding({
        left: '16lpx',
        right: '16lpx'
      })
    }
  }
  // 标题
  @Builder
  TitleBar(title: string) {
    // 标题
    Row() {
      Text(title)
        .fontWeight(FontWeight.Bold)
        .fontColor(Color.Black)
        .fontSize('36lpx')
      Text('更多')
        .fontColor(Color.Gray)
        .onClick(() => {
          if (title=='推荐歌单') {
            router.pushUrl({url:'pages/AllSongList'})
          }else{
            router.pushUrl({url:'pages/AllSingerList'})
          }
        })
    }
    .width('100%')
    .height('100lpx')
    .padding('16lpx')
    .justifyContent(FlexAlign.SpaceBetween)
  }
  // 搜索框
  @Builder
  SearchInput() {
    Row() {
      Row({ space: '8lpx' }) {
        Image($r('app.media.ic_search'))
          .width('40lpx')
          .aspectRatio(1)
          .fillColor('#6a828e')
        Text('生如夏花🔥')
          .fontColor('#6a828e')
          .layoutWeight(1)
        Image($r('app.media.ic_code'))
          .width('40lpx')
          .aspectRatio(1)
          .fillColor('#6a828e')
      }
      .width('100%')
      .padding('24lpx')
      .borderRadius('60')
      .backgroundColor('#cccccc')
      .onClick(()=> {
        router.pushUrl({
          url: "pages/Search1"
        })
      })
    }
    .width('100%')
    .padding('24lpx')
  }
  build() {
    Column() {
      Scroll(){
        Column() {
          this.SearchInput()
          this.getSwiperListWidget()
          this.getSongListWidget()
          this.getSingerWidget()
        }
      }
    }
    .width('100%')
    .height('100%')
  }
}