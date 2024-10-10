/******/ (function() { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ 262:
/***/ (function(module, __unused_webpack_exports, __webpack_require__) {

var $app_template$ = __webpack_require__(236)
var $app_style$ = __webpack_require__(431)
var $app_script$ = __webpack_require__(602)
var options=$app_script$
 if ($app_script$.__esModule) {

      options = $app_script$.default;
 }
options.styleSheet=$app_style$
options.render=$app_template$;
module.exports=new ViewModel(options);

/***/ }),

/***/ 431:
/***/ (function(module) {

module.exports = {"classSelectors":{"container":{"flexDirection":"column","width":"100%","height":"100%","justifyContent":"center","alignItems":"center"},"text":{"width":400,"fontSize":30,"textAlign":"center"}}}

/***/ }),

/***/ 236:
/***/ (function(module) {

module.exports = function (vm) { var _vm = vm || this; return _c('div', {'staticClass' : ["container"]} , [_c('text', {'attrs' : {'value' : "悲伤的燏某人"},'staticClass' : ["text"]} ),_c('text', {'attrs' : {'value' : function () {return decodeURI('%E6%AD%A5%E6%95%B0%EF%BC%9A')+ decodeURI('') +(_vm.steps)}},'staticClass' : ["text"]} ),_c('text', {'attrs' : {'value' : function () {return decodeURI('%E5%BF%83%E7%8E%87%EF%BC%9A')+ decodeURI('') +(_vm.heartRate)}},'staticClass' : ["text"]} )] ) }

/***/ }),

/***/ 602:
/***/ (function(__unused_webpack_module, exports, __webpack_require__) {

"use strict";


var _interopRequireDefault = __webpack_require__(143);
Object.defineProperty(exports, "__esModule", ({
  value: true
}));
exports["default"] = void 0;
var _system = _interopRequireDefault(requireModule("@system.sensor"));
var _default = {
  data: {
    steps: 0,
    heartRate: 0
  },
  onInit: function onInit() {
    var that = this;
    _system["default"].subscribeStepCounter({
      success: function success(ret) {
        that.steps = ret.steps;
      },
      fail: function fail(data, code) {
        console.log('Subscription failed. Code: ' + code + '; Data: ' + data);
      }
    });
    _system["default"].subscribeHeartRate({
      success: function success(ret) {
        that.heartRate = ret.heartRate;
      },
      fail: function fail(data, code) {
        console.log('Subscription failed. Code: ' + code + '; Data: ' + data);
      }
    });
  },
  onDestroy: function onDestroy() {
    _system["default"].unsubscribeStepCounter();
    _system["default"].unsubscribeHeartRate();
  }
};
exports["default"] = _default;

function requireModule(moduleName) {
  return requireNative(moduleName.slice(1));
}


/***/ }),

/***/ 143:
/***/ (function(module) {

"use strict";


function _interopRequireDefault(obj) {
  return obj && obj.__esModule ? obj : {
    "default": obj
  };
}
module.exports = _interopRequireDefault, module.exports.__esModule = true, module.exports["default"] = module.exports;

function requireModule(moduleName) {
  return requireNative(moduleName.slice(1));
}


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module is referenced by other modules so it can't be inlined
/******/ 	var __webpack_exports__ = __webpack_require__(262);
/******/ 	
/******/ 	return __webpack_exports__;
/******/ })()
;
//# sourceMappingURL=index.js.map