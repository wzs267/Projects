import router from '@ohos.router'
import { ConfigUtil as Config } from './common/ConfigUtil'
import { searchSong } from './common/Api'
import { SongListModel, SongModel, searchCateNameModel } from './model/AppModel'
import { ParamModel } from './model/paramModel'
import { PullToRefresh} from '@ohos/pulltorefresh'  //  ohpm install @ohos/pulltorefresh@2.1.0
import { SubHeaderItemWidget } from './widget/SubHeaderItemWidget'
import searchService from './services/SearchService'
import inputMethod from '@ohos.inputMethod'
import { speechRecognizer } from '@kit.CoreSpeechKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { ICapturerInterface } from './utils/ICapturerInterface';
import FileCapturer from './utils/FileCapturer';
import AudioCapturer from './utils/AudioCapturer'
let asrEngine: speechRecognizer.SpeechRecognitionEngine;
@Entry
@Component
struct songList {
  @State sessionId: string = "123456";
  @State isFlag: boolean = false
  private mFileCapturer: ICapturerInterface = new FileCapturer();
  private mAudioCapturer: ICapturerInterface = new AudioCapturer();
  @State type: number = 1
  @State cid: string = ""
  @State keywords: string = ""
  @State songList: SongListModel[] = []
  private scroller: Scroller = new Scroller();
  @State page: number = 1
  @State hasMore: boolean = true
  @State sort: string = ""
  @State selectedSubHeaderIndex: number = 0
  @State subHeaderList: searchCateNameModel[] = [
    new searchCateNameModel(1, "歌曲", 1,),
    new searchCateNameModel(2, "歌手", 2,),
    new searchCateNameModel(3, "歌单", 3,),
  ];
  aboutToAppear() {
    const params:ParamModel=  (router.getParams() as ParamModel)
    this.keywords =params.keywords?params.keywords:""
    //this.getSongListData()
  }
  async getSongListData(resolve?: (value: string | PromiseLike<string>) => void) {
    let params: ParamModel
    params = {
      type:this.type ,
      keywords: this.keywords,
      page: this.page
    }
    try {
      const res = await searchSong(params)
      let tempData: SongListModel[] = JSON.parse(JSON.stringify(res.data)).list
      this.songList = this.songList.concat(tempData)
      this.page++
      if (tempData.length < 3) {
        this.hasMore = false
      }
    } catch (err) {
      console.error("获取数据失败:", err)
    }
    if (resolve) {
      resolve("")
    }
  }
  //筛选数据(分页)
  changeSubHeader(key: number) {
    this.selectedSubHeaderIndex = key
    //重置数据
    this.songList = []
    this.page = 1
    this.hasMore = true
    this.type = this.subHeaderList[key].type
    //重新请求数据
    this.getSongListData()
  }

  //顶部导航和状态栏
  @Builder
  AppBarWidget() {
    Column() {
      //导航
      Row() {
        Row() {
          Image($r("app.media.back")).height('44lpx').padding({ left: '12lpx'}).fillColor(Color.Gray)
        }.width('40lpx')
        .height('100%')
        .onClick(() => {
          router.back()
        })

        Row() {
          Search({
            value: this.keywords,
            placeholder: '查询 歌曲 ,歌单'
          })
            .height('96lpx')
            .backgroundColor('#FCF3EC')
            .placeholderColor(Color.Grey)
            .placeholderFont({ size: "38lpx", weight: FontWeight.Normal})
            .textFont({ size: "38lpx", weight: FontWeight.Normal })
            .margin({
              left: "44lpx",
              right: "44lpx"
            })
            .onChange(async (value) => {
              //执行搜索
              this.keywords = value
              //保存搜索关键词
              if (this.keywords != '') {
                await searchService.setHistoryData(value)
              }
              //重置数据
              this.songList = []
              this.page = 1
              this.hasMore = true
              if (this.selectedSubHeaderIndex==0) {
                this.getSongListData()
              }

              //隐藏键盘
              let im = inputMethod.getController()
              im.stopInputSession()
            })
        }.layoutWeight(1)
        // 语音识别按钮
        Row() {
          //
          Image($r('app.media.huatong'))
            .fillColor(this.isFlag ? Color.Orange : Color.Gray)
            .width('40lpx')
            .height('40lpx')
            .margin({ right: '20lpx' })
            .onTouch(async (event: TouchEvent) => {
              if (event.type === TouchType.Down) {
                this.isFlag = true;
                await this.startRecording();
              }
              if (event.type === TouchType.Up) {
                setTimeout(() => {
                  this.isFlag = false;
                  asrEngine.finish(this.sessionId);
                }, 1000);
              }
            })
        }
      }.width("100%").height("140lpx")
      .padding({
        left: "34lpx",
        right: "34lpx"
      }).justifyContent(FlexAlign.SpaceBetween)
    }
    .backgroundColor(Color.White)
  }

  //列表
  @Builder
  songListWidget() {
    Scroll(this.scroller) {
      Column({
        space: "20lpx"
      }) {
        ForEach(this.songList, (item: SongModel) => {
          Row() {
            Image(Config.SERVER_URL + item.pic).width("260lpx").margin({ right: "34lpx" })
            Column() {
              if (this.type== 3) {
                Text(item.title).fontSize("44lpx").fontWeight(FontWeight.Normal)
              }else{
                Text(item.name).fontSize("44lpx").fontWeight(FontWeight.Normal)
              }
              Text(`${item.introduction}`)
                .width('90%')
                .fontSize("36lpx")
                .maxLines(3) // 最多显示2行
                .textOverflow({ overflow: TextOverflow.Ellipsis }) // 超出部分显示省略号
            }
            .layoutWeight(1)
            .alignItems(HorizontalAlign.Start)
            .justifyContent(FlexAlign.SpaceEvenly)
            .height("100%")
          }
          .width("100%")
          .height("380lpx")
          .borderRadius("15lpx")
          .padding("20lpx")
          .onClick(() => {
            console.log("点击了"+ item.id);
            if (this.type == 1){
              router.pushUrl({
                url: 'pages/SongsDetail',
                params: {
                  cid: item.id
                }
              })
            }else if (this.type == 2) {
              router.pushUrl({
                url: 'pages/SingerDetail',
                params: {
                  cid: item.id
                }
              })
            } else if (this.type == 3) {
              router.pushUrl({
                url: 'pages/SongListDetail',
                params: {
                  cid: item.id
                }
              })
            }
          })
        })
        if (!this.hasMore) {
          Text("-----我是有底线的-----").width("100%").margin("10lpx").textAlign(TextAlign.Center)
        }
      }.width("100%")
      .padding("20lpx")
    }.width("100%").edgeEffect(EdgeEffect.None)
  }

  //二级导航
  @Builder
  SubHeaderWidget() {
    Row() {
      ForEach(this.subHeaderList, (item: searchCateNameModel, key: number) => {
        SubHeaderItemWidget({
          item: item,
          index: key,
          selectedSubHeaderIndex: $selectedSubHeaderIndex, //@Link需要注意写法
          func: (index) => { //让子组件执行父组件的方法
            this.changeSubHeader(index)
          }
        })
      })

    }.width("100%")
    .height("120lpx")
    .backgroundColor(Color.White)
  }
  build() {
    Stack({
      alignContent: Alignment.Top
    }) {
      Column() { //内容区域
        PullToRefresh({
          // 必传项，需绑定传入主体布局内的列表或宫格组件
          scroller: this.scroller,
          // 必传项，自定义主体布局，内部有列表或宫格组件
          customList: () => {
            // 一个用@Builder修饰过的UI方法
            this.songListWidget()
          },
          onLoadMore: () => {
            return new Promise<string>((resolve) => {
              this.getSongListData(resolve)
            })
          }
        })
      }
      .width('100%')
      .height('100%')
      .backgroundColor("#F4F4F4")
      .padding({
        top: "250lpx"
      })

      Column() { //头部
        this.AppBarWidget()
        this.SubHeaderWidget()
      }.width('100%')
    }

  }
  // Setting callbacks
  setListener() {
    // Creating a Callback Object
    let setListener: speechRecognizer.RecognitionListener = {
      // Callback for successful recognition start
      onStart: (sessionId: string, eventMessage: string) => {
        this.keywords = '';
        console.info("setListener onStart sessionId: " + sessionId + "eventMessage: " + eventMessage);
      },
      // Event callback
      onEvent(sessionId: string, eventCode: number, eventMessage: string) {
        console.info("setListener onEvent sessionId: " + sessionId + "eventMessage: " + eventCode + "eventMessage: " + eventMessage);
      },
      // Identification result callback, including intermediate results and final results.
      onResult: (sessionId: string, res: speechRecognizer.SpeechRecognitionResult) => {
        let isFinal: boolean = res.isFinal;
        let isLast: boolean = res.isLast;
        let result: string = res.result;
        let regexp: RegExp = new RegExp(/\。$/, "g");
        result = result.replaceAll(regexp, "");
        this.keywords = result;
        console.info('setListener onResult: ' + 'sessionId: ' + sessionId + ' isFinal: ' + isFinal + ' isLast: ' + isLast + ' result: ' + result);
      },
      // Recognition completion callback
      onComplete(sessionId: string, eventMessage: string) {
        console.info("setListener onComplete sessionId: " + sessionId + "eventMessage: " + eventMessage);
      },
      // Error callback. The error code is returned by this method.
      onError(sessionId: string, errorCode: number, errorMessage: string) {
        console.error("setListener onError sessionId: " + sessionId + "errorCode: " + errorCode + "errorMessage: " + errorMessage);
      }
    }
    // Invoke the callback method.
    asrEngine.setListener(setListener);
  };
  // Recording to text
  async startRecording() {
    // MLRemoteAftSetting setting = new MLRemoteAftSetting.Factory()
    //   .enablePunctuation(false)
    //   .create();
    let extraParam: Record<string, Object> = {
      "locate": "CN",
      "recognizerMode": "short",
    };
    //speechRecognizer.CreateEngineParams
    let initParamsInfo: speechRecognizer.CreateEngineParams = {
      language: 'zh-CN',
      online: 1,
      extraParams: extraParam
    };

    // Invoke the createEngine method.
    speechRecognizer.createEngine(initParamsInfo)
      .then((speechRecognitionEngine: speechRecognizer.SpeechRecognitionEngine) => {
        asrEngine = speechRecognitionEngine;
        this.setListener();
        // Set the parameters related to the start of identification.
        let audioParam: speechRecognizer.AudioInfo = { audioType: 'pcm', sampleRate: 16000, soundChannel: 1, sampleBit: 16 }
        let extraParam: Record<string, Object> = {
          "recognitionMode": 0,
          "vadBegin": 2000,
          "vadEnd": 3000,
          "maxAudioDuration": 20000
        }
        let recognizerParams: speechRecognizer.StartParams = {
          sessionId: this.sessionId,
          audioInfo: audioParam,
          extraParams: extraParam
        }
        // Invoke the start recognition method.
        console.info('startListening start');
        asrEngine.startListening(recognizerParams);

        // Recording Obtaining Audio
        let data: ArrayBuffer;
        this.mFileCapturer = this.mAudioCapturer;
        console.info('create capture success');
        this.mFileCapturer.init((dataBuffer: ArrayBuffer) => {
          console.info('start write');
          console.info('ArrayBuffer ' + JSON.stringify(dataBuffer));
          data = dataBuffer
          let uint8Array: Uint8Array = new Uint8Array(data);
          console.info('ArrayBuffer uint8Array ' + JSON.stringify(uint8Array));
          // Writing Audio Streams
          asrEngine.writeAudio(this.sessionId, uint8Array);
        });
      })
      .catch((err: BusinessError) => {
        console.error('result' + JSON.stringify(err));
      });

  };
}