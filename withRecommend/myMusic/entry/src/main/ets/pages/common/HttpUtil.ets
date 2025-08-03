import http from '@ohos.net.http'
import { ResponseModel} from "../model/paramModel"
import { ConfigUtil as Config } from "./ConfigUtil"
import preferencesUtil from '../common/PreferencesUtil'

// 向 header中添加 authorization: 'token加密'
async function addTokenToHeader(options: http.HttpRequestOptions) {
  // 从存储中获取 Token（实际项目中可能从偏好设置或其他安全存储中获取）
  const token = await preferencesUtil.get("userinfoStore", "token", "") || ''
  console.log('=========',token)
  if (token) {
    // 确保 header 对象存在
    if (!options.header) {
      options.header = {}
    }
    // 添加 Token 到请求头，使用 Bearer 格式
    options.header['authorization'] = `${token}`
  }
  return options
}

export async function HttpGet(apiUri: string): Promise<ResponseModel> {
  let httpReq: http.HttpRequest = http.createHttp()
  // 构建请求选项并添加 Token
  const options = await addTokenToHeader({
    method: http.RequestMethod.GET,
    connectTimeout: Config.HTTP_CONNECT_TIMEOUT,
    readTimeout: Config.HTTP_READ_TIMEOUT,
  })
  let promiseData: Promise<http.HttpResponse> = httpReq.request(Config.SERVER+apiUri, options)
  let serverData: ResponseModel = new ResponseModel()
  return promiseData.then((response) => {
    httpReq.destroy() // 关闭请求，释放资源
    if (response.responseCode == Config.HTTP_CODE_200) {
      let data = `${response.result}` //把json对象转换成json字符串
      let result: object = JSON.parse(data) //把json字符串转换成json对象
      serverData.success = true
      serverData.message = result["message"] ||  Config.SERVER_NET_SUCCESS
      serverData.data = result["data"]
      return serverData
    } else {
      serverData.message = Config.SERVER_NET_ERROR
      return serverData
    }
  }).catch((e: object) => {
    httpReq.destroy() // 关闭请求，释放资源
    serverData.message = `${Config.SERVER_NET_ERROR}:${JSON.stringify(e)}`;
    return serverData;
  })
}


export async function  HttpPost(apiUri: string, extraData?: object): Promise<ResponseModel> {
  let httpReq: http.HttpRequest = http.createHttp()
  // 构建请求选项并添加 Token
  const options = await addTokenToHeader({
    method: http.RequestMethod.POST,
    header: {
      'Content-Type': 'application/json', // json数据

    },
    extraData: extraData,
    expectDataType: http.HttpDataType.OBJECT,  // 返回对象类型
    connectTimeout: Config.HTTP_CONNECT_TIMEOUT,
    readTimeout: Config.HTTP_READ_TIMEOUT
  })
  let promiseData: Promise<http.HttpResponse> = httpReq.request(Config.SERVER+apiUri,options)

  let serverData: ResponseModel = new ResponseModel()
  return promiseData.then((response) => {
    httpReq.destroy() // 关闭请求，释放资源
    if (response.responseCode == Config.HTTP_CODE_200) {
      let data = `${JSON.stringify(response.result)}` //把json对象转换成json字符串
      let result: object = JSON.parse(data) //把json字符串转换成json对象
      serverData.success = true
      serverData.message = result["message"]
      serverData.data = result['data']
      return serverData
    } else {
      serverData.message = Config.SERVER_NET_ERROR
      return serverData
    }
  }).catch((e: object) => {
    httpReq.destroy() // 关闭请求，释放资源
    serverData.message = `${Config.SERVER_NET_ERROR}:${JSON.stringify(e)}`;
    return serverData;
  })
}

