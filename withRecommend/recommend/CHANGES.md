# AI推荐系统集成修改清单

## 修改概述
将前端项目首页的推荐歌单改为根据当前Recommend模型训练结果推荐单曲，实现AI驱动的个性化音乐推荐。

---

## 🔧 后端修改 (Node.js/Koa)

### 1. `openKoa/controller/api.js`
**新增方法：** `getRecommendSongs`
```javascript
// 新增AI推荐单曲接口
getRecommendSongs: async (ctx) => {
  const { userId = 3, count = 5 } = ctx.query;
  
  try {
    // 执行Python推荐脚本
    const result = await new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ["scripts/api/simple_recommend.py", userId, count], {
        cwd: path.join(__dirname, '../../recommend'),
        stdio: ['pipe', 'pipe', 'pipe']
      });
      
      // 处理Python脚本输出和错误处理
      // 支持TensorFlow警告的容错处理
      // JSON解析即使在非零退出码情况下
    });
    
    // 获取歌曲详细信息并合并推荐分数
    const songDetails = await Song.findAll({
      where: { id: result.recommendations.map(rec => rec.song_id) },
      include: [{ model: Singer, as: 'singer' }]
    });
    
    // 返回AI推荐结果
    ctx.body = {
      code: 200,
      message: "AI推荐单曲成功",
      data: detailedRecommendations,
      count: detailedRecommendations.length
    };
    
  } catch (error) {
    // 降级方案：返回默认歌曲列表
  }
}
```

### 2. `openKoa/routes/index.js`
**新增路由：**
```javascript
router.get('/getRecommendSongs', controller.getRecommendSongs)
```

---

## 🎯 前端修改 (HarmonyOS/ArkTS)

### 3. `myMusic/entry/src/main/ets/pages/model/AppModel.ets`
**新增数据模型：**
```typescript
// AI推荐歌曲模型
export interface RecommendSongModel {
  id: number;
  name: string;
  pic: string;
  url: string;
  introduction?: string;
  recommendScore?: number; // 推荐分数
  reason?: string; // 推荐理由
  singer?: SingerModel; // 歌手信息
}
```

### 4. `myMusic/entry/src/main/ets/pages/common/Api.ets`
**新增API接口：**
```typescript
// AI推荐单曲接口
export const getRecommendSongs = (userId?: number, count?: number)=> {
  let params = '';
  if (userId || count) {
    const paramList: string[] = [];
    if (userId) paramList.push(`userId=${userId}`);
    if (count) paramList.push(`count=${count}`);
    params = '?' + paramList.join('&');
  }
  return HttpGet(`/getRecommendSongs${params}`);
}
```

### 5. `myMusic/entry/src/main/ets/pages/tabs/Home.ets`
**主要修改：**

#### 状态变量修改：
```typescript
// 新增
@State recommendSongs:RecommendSongModel[] = []
// 修改类型
@StorageLink("userInfo") userInfo: Record<string, ESObject> = {}
```

#### 推荐数据获取方法重构：
```typescript
async recommendSongListData(){
  try {
    // 优先使用AI推荐
    const userId = (this.userInfo?.user?.id as number) || 3;
    const res = await getRecommendSongs(userId, 6);
    const resRecord = res as Record<string, ESObject>;
    const isSuccess = resRecord.code === 200 || res.success;
    const responseData: ESObject = resRecord.data || res.data;
    
    if (isSuccess && responseData) {
      this.recommendSongs = responseData as RecommendSongModel[];
    } else {
      // 降级到原有歌单推荐
    }
  } catch (error) {
    // 错误处理和降级逻辑
  }
}
```

#### UI组件完全重构：
- **从歌单网格布局改为单曲列表布局**
- **新增播放方法：** `playRecommendSong(song: RecommendSongModel)`
- **播放按钮改为文本符号：** `Text("▶")` 替代缺失的图标资源

---


## ✅ 集成结果

1. **✅ 后端AI推荐API** - `/getRecommendSongs` 接口正常工作
2. **✅ Python脚本集成** - 解决了Windows GBK编码问题
3. **✅ 前端UI重构** - 从歌单推荐改为单曲推荐显示
4. **✅ 降级机制** - AI推荐失败时自动降级到默认歌曲列表
5. **✅ 类型安全** - 修复了HarmonyOS ArkTS编译错误

## 🔄 工作流程

1. 前端调用 `getRecommendSongs(userId, count)` 
2. 后端执行 `python scripts/api/simple_recommend.py userId count`
3. Python返回JSON格式的推荐结果（歌曲ID + 推荐分数）
4. 后端查询数据库获取歌曲详细信息
5. 前端展示AI推荐的单曲列表

## 📋 测试状态

- **API测试：** ✅ `curl "http://localhost:3000/getRecommendSongs?userId=3&count=3"` 返回AI推荐结果
- **Python脚本：** ✅ 直接执行返回正确JSON格式
- **前端编译：** ✅ 解决了所有ArkTS类型错误
- **降级机制：** ✅ AI失败时正确降级到默认推荐

---

*修改完成时间：2025年7月22日*
*修改人：GitHub Copilot*
*感谢Copilot的帮助*
