<script setup lang="ts">
import { computed, defineAsyncComponent, ref, shallowRef, watchEffect } from 'vue'

// 1. 自动扫描 src/components/dataset 下的所有 .vue 文件
// eager: false 表示懒加载，只有用到时才加载
const modules = import.meta.glob('./components/dataset/*.vue')

const params = new URLSearchParams(window.location.search)
const componentName = params.get('component') // 例如: 'Accordion_001'

// 2. 动态组件引用
const currentComponent = shallowRef<any>(null)
const errorMsg = ref('')

watchEffect(async () => {
  if (!componentName) {
    errorMsg.value = 'No component specified in URL'
    return
  }

  const path = `./components/dataset/${componentName}.vue`
  
  if (path in modules) {
    // 动态导入组件
    const mod: any = await modules[path]()
    currentComponent.value = mod.default
  } else {
    errorMsg.value = `Component not found: ${componentName}`
  }
})
</script>

<template>
  <div id="playground-root" class="w-full min-h-screen flex items-center justify-center bg-white">
    <!-- 动态渲染组件 -->
    <component :is="currentComponent" v-if="currentComponent" />
    
    <!-- 错误提示 -->
    <div v-else class="text-red-500 font-bold">{{ errorMsg }}</div>
  </div>
</template>