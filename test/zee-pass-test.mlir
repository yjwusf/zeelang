// RUN: zee-opt --zee-switch-bar-foo %s | FileCheck %s

// CHECK-LABEL: func @foo()
func.func @bar() {
  return
}
