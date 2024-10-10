/******/ (function() { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ 712:
/***/ (function(module, __unused_webpack_exports, __webpack_require__) {

var $app_script$ = __webpack_require__(844)
var options=$app_script$
 if ($app_script$.__esModule) {

        options = $app_script$.default;
 }
module.exports=new ViewModel(options);

/***/ }),

/***/ 844:
/***/ (function(module, exports, __webpack_require__) {

"use strict";


Object.defineProperty(exports, "__esModule", ({
  value: true
}));
exports["default"] = void 0;
var _default = {
  onCreate: function onCreate() {
    console.info('Application onCreate');
  },
  onDestroy: function onDestroy() {
    console.info('Application onDestroy');
  }
};
exports["default"] = _default;
;
(exports["default"] || module.exports).manifest = __webpack_require__(378);

function requireModule(moduleName) {
  return requireNative(moduleName.slice(1));
}


/***/ }),

/***/ 378:
/***/ (function(module) {

"use strict";
module.exports = JSON.parse('{"manifest.json":"content"}');

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
/******/ 	var __webpack_exports__ = __webpack_require__(712);
/******/ 	
/******/ 	return __webpack_exports__;
/******/ })()
;
//# sourceMappingURL=app.js.map