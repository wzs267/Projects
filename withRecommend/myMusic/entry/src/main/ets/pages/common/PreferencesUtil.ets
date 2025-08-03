import preferences from '@ohos.data.preferences';
//  API 持久化
class PreferencesUtil {
  //设置数据  PreferencesUtil.set('userStore','userIfno',{user:"张'，age:18})
  async set(storeName: string, key: string, value: preferences.ValueType) {
    try {
      let context: Context = getContext(this)  // 获取上下文对象
      let pref = await preferences.getPreferences(context, storeName)  // preferences.getPreferences 获取Preferences实例
      //写入数据
      await pref.put(key, value)
      //flush将Preferences实例持久化
      await pref.flush()
    } catch (e) {

      console.log('testTag', `保存preferences失败`, JSON.stringify(e));
    }
  }

  //获取数据
  async get(storeName: string, key: string, defValue: preferences.ValueType): Promise<preferences.ValueType>{
    try {
      let context: Context = getContext(this)
      let pref = await preferences.getPreferences(context, storeName)
      //获取数据
      let data = await pref.get(key, defValue)
      return data
    } catch (e) {
      console.log('testTag', `获取数据失败`, JSON.stringify(e));
      return defValue
    }
  }

  //删除数据
  async delete(storeName: string, key: string) {
    try {
      let context: Context = getContext(this)
      let pref = await preferences.getPreferences(context, storeName)
      //删除数据
      await pref.delete(key)
    } catch (e) {
      console.log('testTag', `删除数据`, JSON.stringify(e));
    }
  }
}

let preferencesUtil = new PreferencesUtil()

export default preferencesUtil