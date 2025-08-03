import router from '@ohos.router'
import searchService from './services/SearchService'
import promptAction from '@ohos.promptAction'
import { speechRecognizer } from '@kit.CoreSpeechKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { ICapturerInterface } from './utils/ICapturerInterface';
import FileCapturer from './utils/FileCapturer';
import AudioCapturer from './utils/AudioCapturer'

let asrEngine: speechRecognizer.SpeechRecognitionEngine;

@Entry
@Component
struct Search1 {
  @State historySearch: string[] = []
  @State hotSearch: string[] = ['愿望', '青花瓷', '明天']
  @State keywords: string = ""
  @State isFlag: boolean = false
  @State sessionId: string = "123456";
  private mFileCapturer: ICapturerInterface = new FileCapturer();
  private mAudioCapturer: ICapturerInterface = new AudioCapturer();

  aboutToAppear() {
    this.getHistoryData() //获取历史记录
  }

  //获取历史记录
  async getHistoryData() {
    this.historySearch = await searchService.getHistoryData()
  }

  //保存历史记录执行跳转
  async doSearch() {
    if (this.keywords != "") {
      await searchService.setHistoryData(this.keywords)
    }
    //替换路由
    router.replaceUrl({
      url: 'pages/SearchList',
      params: {
        "keywords": this.keywords,
        "cid": ""
      }
    })
  }

  //自定义组件必须使用根组件来包裹
  @Builder
  AppBarWidget() {
    Column() {
      //导航
      Row() {
        Row() {
          Image($r("app.media.back")).height("50lpx").padding({ left: "6lpx" })
        }.width("40lpx")
        .height('100%')
        .onClick(() => {
          router.back()
        })

        Row() {
          Search({
            value: this.keywords,
            placeholder: '查歌曲'
          })
            .height('96lpx')
            .backgroundColor('#cccccc')
            .placeholderColor('#999999')
            .placeholderFont({ size: "38lpx", weight: FontWeight.Normal })
            .textFont({ size: "38lpx", weight: FontWeight.Normal })
            .onChange((value: string) => {
              this.keywords = value
            })
            .margin({
              left: "44lpx",
              right: "44lpx"
            })
        }.layoutWeight(1)

        // 语音识别按钮
        Row() {
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
        Row() {
          Text("搜索").fontSize("38lpx")
        }.width("100lpx").onClick(() => {
          this.doSearch()
        })
      }.width("100%").height("140lpx")
      .padding({
        left: "34lpx",
        right: "34lpx"
      }).justifyContent(FlexAlign.SpaceBetween)
    }
  }

  build() {
    Column() {
      this.AppBarWidget()
      Scroll() {
        Column() {
          if (this.historySearch.length > 0) {
            Row() {
              Text("搜索历史").padding({ left: "28lpx" }).fontSize('44lpx').fontWeight(FontWeight.Bold)
              Image($r("app.media.delete1")).width("70lpx").padding({ right: "28lpx" }).onClick(async () => {

                promptAction.showDialog({
                  title: "提示信息!",
                  message: '您确定要清空历史记录吗？',
                  buttons: [
                    {
                      text: '取消',
                      color: '#FF0000'
                    },
                    {
                      text: '确认',
                      color: '#0099FF'
                    }
                  ]
                }).then(async (result) => {
                  if (result.index === 1) {
                    await searchService.cleanHistoryData()
                    this.getHistoryData() // 再次获取历史记录
                  }
                })

              })
            }.width("100%")
            .justifyContent(FlexAlign.SpaceBetween)
            .margin({ top: "40lpx", bottom: "20lpx" })

            Flex({
              wrap: FlexWrap.Wrap
            }) {
              ForEach(this.historySearch, (item: string) => {
                Button({
                  type: ButtonType.Normal
                }) {
                  Text(item).fontSize("34lpx")
                }
                .padding({
                  left: "34lpx",
                  right: "34lpx",
                  top: "14lpx",
                  bottom: "14lpx"
                })
                .margin("20lpx")
                .backgroundColor("#eee")
                .borderRadius("14lpx")
                .onClick(() => {
                  this.keywords = item
                })
                .gesture(
                  LongPressGesture(
                    { duration: 1000 }
                  ).onAction(() => {
                    promptAction.showDialog({
                      title: "提示信息!",
                      message: '您确定要删除吗？',
                      buttons: [
                        {
                          text: '取消',
                          color: '#FF0000'
                        },
                        {
                          text: '确认',
                          color: '#0099FF'
                        }
                      ]
                    }).then(async (result) => {
                      if (result.index === 1) {
                        await searchService.deleteHistoryData(item)
                        this.getHistoryData()
                      }
                    })
                  })
                )
              })
            }
          }

          Row() {
            Text("猜你想搜").padding({ left: "28lpx" }).fontSize('44lpx').fontWeight(FontWeight.Bold)
          }.width("100%")
          .justifyContent(FlexAlign.SpaceBetween)
          .margin({ top: "40lpx", bottom: "20lpx" })

          Flex({
            wrap: FlexWrap.Wrap
          }) {
            ForEach(this.hotSearch, (item: string) => {
              Button({
                type: ButtonType.Normal
              }) {
                Text(item).fontSize("34lpx")
              }
              .padding({
                left: "34lpx",
                right: "34lpx",
                top: "14lpx",
                bottom: "14lpx"
              })
              .margin("20lpx")
              .backgroundColor("#eee")
              .borderRadius("14lpx")
              .onClick(() => {
                this.keywords = item
              })
            })
          }

        }.width("100%")
      }
    }
    .width('100%')
    .height('100%')
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
        console.info("setListener onEvent sessionId: " + sessionId + "eventMessage: " + eventCode + "eventMessage: " +
          eventMessage);
      },
      // Identification result callback, including intermediate results and final results.
      onResult: (sessionId: string, res: speechRecognizer.SpeechRecognitionResult) => {
        let isFinal: boolean = res.isFinal;
        let isLast: boolean = res.isLast;
        let result: string = res.result;
        let regexp: RegExp = new RegExp(/\。$/, "g");
        result = result.replaceAll(regexp, "");
        this.keywords = result;
        console.info('setListener onResult: ' + 'sessionId: ' + sessionId + ' isFinal: ' + isFinal + ' isLast: ' +
          isLast + ' result: ' + result);
      },
      // Recognition completion callback
      onComplete(sessionId: string, eventMessage: string) {
        console.info("setListener onComplete sessionId: " + sessionId + "eventMessage: " + eventMessage);
      },
      // Error callback. The error code is returned by this method.
      onError(sessionId: string, errorCode: number, errorMessage: string) {
        console.error("setListener onError sessionId: " + sessionId + "errorCode: " + errorCode + "errorMessage: " +
          errorMessage);
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
        let audioParam: speechRecognizer.AudioInfo = {
          audioType: 'pcm',
          sampleRate: 16000,
          soundChannel: 1,
          sampleBit: 16
        }
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
