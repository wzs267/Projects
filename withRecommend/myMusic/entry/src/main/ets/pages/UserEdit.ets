import router from '@ohos.router'
import { ConfigUtil as Config } from './common/ConfigUtil'
import { UserModel } from './model/AppModel'
import {UserEditParam} from './model/paramModel'
import { CityPicker } from './widget/CityPicker'
import promptAction from '@ohos.promptAction'
import inputMethod from '@ohos.inputMethod'
import {getUserInfo,userEdit} from './common/Api'
import { BusinessError, request } from '@kit.BasicServicesKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import fs from '@ohos.file.fs';
import preferencesUtil from './common/PreferencesUtil'

@Entry
@Component
struct UserEdit {
  @State pickerWidgetTranslate: number = lpx2px(600)
  @StorageLink("userInfo") userInfo: UserModel = {}
  @State message: string = 'Hello World';
  @State imagePath: string = ""
  @State userData :UserEditParam ={
    avatar:this.userInfo.avatar,
    sex:this.userInfo.sex,
    phone_num:this.userInfo.phone_num ,
    introduction:this.userInfo.introduction,
    location:this.userInfo.location,
  }
  async getUserInfoData(){
    const response  = await getUserInfo()
    if (response.success) {
      let data = JSON.stringify(response.data)
      console.log("获取用户信息：" + data)
      let user: string = JSON.stringify(response.data ? response.data["user"] : "")
      await preferencesUtil.set("userinfoStore", "userinfo", user)
      this.userInfo = response.data ? response.data["user"] : ""
      //PersistentStorage.persistProp("userInfo", this.userInfo)
    }
  }
  async userEditData() {
    //校验表单是否合法
    if (this.userData.avatar == "") {
      promptAction.showToast({
        message: "头像不能为空"
      })
      return;
    } else if (this.userData.phone_num == "") {
      promptAction.showToast({
        message: "手机号不能为空"
      })
      return;
    } else if (this.userData.location == "") {
      promptAction.showToast({
        message: "地址不能为空"
      })
      return;
    } else if (this.userData.introduction == "") {
      promptAction.showToast({
        message: "座右铭不能为空"
      })
      return;
    }
    const response  =await userEdit(this.userData)
    if (response.success) {
      this.getUserInfoData()   // 获取用户信息接口
    }

  }
  pickerPhoto = () => {
    try {
      //1、实例化PhotoSelectOptions
      let PhotoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
      //2、配置PhotoView的类型  IMAGE_TYPE 选择图片  VIDEO_TYPE选择视频     IMAGE_VIDEO_TYPE图片和视频
      PhotoSelectOptions.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
      //3、一次最多选择几张图片 或者几个视频
      PhotoSelectOptions.maxSelectNumber = 1;
      //4、获取图片或者视频
      let photoPicker = new photoAccessHelper.PhotoViewPicker();
      photoPicker.select(PhotoSelectOptions).then((PhotoSelectResult: photoAccessHelper.PhotoSelectResult) => {
        console.info('PhotoViewPicker.select successfully, PhotoSelectResult uri: ' +
        JSON.stringify(PhotoSelectResult));
        this.imagePath = PhotoSelectResult.photoUris[0] //重要 （本地选择的图）
        this.convertFileUri(PhotoSelectResult.photoUris[0])
      }).catch((err: BusinessError) => {
        console.error(`PhotoViewPicker.select failed with err: ${err.code}, ${err.message}`);
      });
    } catch (error) {
      let err: BusinessError = error as BusinessError;
      console.error(`PhotoViewPicker failed with err: ${err.code}, ${err.message}`);
    }
  }
  convertFileUri = (uri: string) => {
    //uri  file://media/Photo/296/IMG_1729581568_269/IMG_20241022_151748.jpg
    fs.open(uri, fs.OpenMode.READ_ONLY).then((data) => {

      let context = getContext(this) as Context
      //获取时间戳
      let fileName = Date.now()
      //获取文件后缀名
      let extname = uri.split(".")[1]
      //cacheDir对应的文件路径
      let newDir = context.cacheDir + "/" + fileName + "." + extname
      fs.copyFile(data.fd, newDir).then(() => {
        console.log("newDir:" + newDir) //    /data/storage/el2/base/haps/entry/cache/1729583057537.jpg
        this.doUpload(newDir, extname)
      }).catch((err: Error) => {
        this.message = "失败 -message=" + err.message
      })
    }).catch((err: Error) => {
      this.message = "fsOpen Error message=" + err.message
    })
  }
  doUpload = (newDir: string, extname: string) => {  //上传
    let fileName = newDir.split("cache")[1]
    let uri = "internal://cache" + fileName //  internal://cache/1729583057537.jpg
    console.log("doUploadUri" + uri)
    let uploadTask: request.UploadTask; //监听上传进度
    let uploadConfig: request.UploadConfig = {
      //配置参数
      url: Config.SERVER+'/uploads', //上传文件接口
      header: { 'Accept': '*/*' },
      method: "POST",
      files: [{
        name: 'file', //资源名称
        filename: fileName, //带后缀的文件名
        uri: uri, //本地路径  internal://cache/1729583057537.jpg
        type: extname  //文件类型
      }],
      data: []//[{ name: "uid", value: "123" }, { name: "username", value: "zhangsan" }], //传入的额外的参数
    };
    try {
      request.uploadFile(getContext(), uploadConfig).then((data: request.UploadTask) => {
        uploadTask = data;
        //监听上传进度
        uploadTask.on("progress", (uploadedSize, totalSize) => {
          console.log("upload totalSize:" + totalSize + "  uploadedSize:" + uploadedSize)
        })
        //获取服务器返回的接口数据）
        uploadTask.on('headerReceive', (headers) => {
          console.log('======================================')
          console.log(JSON.stringify(headers["body"])) //{code:200,message:'上传成功',data:"h0wncqAQLRj6wcMLcQnC8HFE.jpg"}
          this.userData.avatar = JSON.parse(headers["body"])['data']  // 重要，获取上传成功后的图片地址 this.userData
        });


      }).catch((err: BusinessError) => {
        console.error(`Failed to request the upload. Code: ${err.code}, message: ${err.message}`);
      });
    } catch (err) {
      console.error(`Failed to request the upload. err: ${JSON.stringify(err)}`);
    }

  }

  //监听页面的销毁事件
  aboutToDisappear() {
  }


  @Builder
  AppBarWidget() {
    Column() {
      //顶部状态栏
      Row().width("100%").height("100lpx")
      //导航
      Row() {
        Row() {
          Image($r("app.media.back")).fillColor(Color.Grey).height("56lpx").padding({ left: "12lpx" })
        }.width("180lpx")
        .height('100%')
        .onClick(() => {
          router.back()
        })

        Text("用户修改")
          .layoutWeight(1)
          .height("140lpx")
          .textAlign(TextAlign.Center)
          .fontSize("35lpx")
          .fontWeight(FontWeight.Bold)

        Row() {
        }.width("180lpx")
      }.width("100%").height("140lpx")
      .padding({
        left: "34lpx",
        right: "34lpx"
      }).justifyContent(FlexAlign.SpaceBetween)
    }
    .backgroundColor(Color.White)
    .position({
      x: 0,
      y: 0
    })

  }

  @Builder
  BodyWidget() {
    Column() {
      Column() {
        Row(){
          if (this.imagePath) {
            Image(this.imagePath)
              .width("148lpx")
              .height("148lpx")
              .borderRadius("74lpx")
              .backgroundColor(Color.Grey)
              .margin({ top: 20 })
          }else{
            Image(`${Config.SERVER_URL}${this.userInfo.avatar}`).width("148lpx").height("148lpx").borderRadius("74lpx")
          }
        }
        .onClick(()=>{
          this.pickerPhoto()
        })
        Row() {
          Text("姓名:")
          Text(this.userInfo.username)
            .layoutWeight(1)
            .margin({ left: "10lpx" })
            .backgroundColor(Color.White)
        }.height("162lpx")
        .borderRadius(0)
        Row(){
          Text("用户性别：").fontSize(16).fontWeight(FontWeight.Bold)
          Radio({ value:"1", group:'sex' })
            .checked(this.userInfo.sex==1) //选中
            .onChange((v)=>{
              if (v==true) {
                this.userData.sex = 1
              }else{
                this.userData.sex = 2
              }
            })
          Text("男")
          Radio({ value:"2", group:'sex' })
            .checked(this.userInfo.sex==2) //选中
          Text("女")
        }
        .justifyContent(FlexAlign.Start)
        .margin(10)
        .width('100%')
        Row() {
          Text("电话:")
          TextInput({
            text: this.userInfo.phone_num,
            placeholder: "会员手机号"
          })
            .layoutWeight(1)
            .type(InputType.Number)
            .margin({ left: "10lpx" })
            .backgroundColor(Color.White)
            .onChange((v) => {
              this.userData.phone_num = v
            })
            .onClick(() => {
              this.pickerWidgetTranslate = lpx2px(600)
            })
        }.height("162lpx")
      }.backgroundColor(Color.White)
      .padding("20lpx")

      Column() {
        Row() {
          Text("所在地区:")
          Row() {
            Text(`${this.userData.location||this.userInfo.location }`).fontColor(Color.Gray).padding({ left: "48lpx" })
          }.layoutWeight(1)
          .onClick(() => {
            this.pickerWidgetTranslate = 0
            //隐藏键盘
            let im = inputMethod.getController()
            im.stopInputSession()
          })

        }.height("162lpx")

        Row() {
          Text("会员介绍:")
          TextInput({
            text: this.userInfo.introduction,
            placeholder: "会员介绍"
          }).layoutWeight(1)
            .margin({ left: "10lpx" })
            .backgroundColor(Color.White)
            .onChange((v) => {
              this.userData.introduction = v
            })
            .onClick(() => {
              this.pickerWidgetTranslate = lpx2px(600)
            })
        }.height("162lpx")

        Button() {
          Text(`修改`).fontColor(Color.White)
        }
        .backgroundColor("#4781c3")
        .width("88%")
        .height("70lpx")
        .onClick(() => {
          this.userEditData()
          router.back()
        })
        .margin({ top: "120lpx" })

      }.backgroundColor(Color.White)
      .margin({
        top: "20lpx"
      }).padding("20lpx")

    }
    .width('100%')
    .height('100%')
    .backgroundColor("#F6F6F6")
    .padding({
      top: "260lpx"
    })
  }

  @Builder
  PickerActionSheet() {
    Column() {

      Row() {
        Stack({
          alignContent: Alignment.TopEnd
        }) {
          Text("选择地址").textAlign(TextAlign.Center).fontSize("40lpx").width("100%")

          Text("完成").textAlign(TextAlign.Center).fontSize("25lpx").width("100lpx").onClick(() => {
            this.pickerWidgetTranslate = lpx2px(600)
          })
        }

      }.width("100%")
      .height("140lpx")
      .backgroundColor(Color.White)
      .justifyContent(FlexAlign.Center)

      CityPicker({
        onChange: (v: string) => {
          console.log(v)
          this.userData.location = v
        }
      })
    }.width("100%")
    .translate({
      y: this.pickerWidgetTranslate
    })
    .backgroundColor(Color.White)

  }

  build() {
    Stack({
      alignContent: Alignment.BottomEnd
    }) {
      this.BodyWidget()
      this.AppBarWidget()
      this.PickerActionSheet()
    }
  }
}