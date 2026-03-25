use super::{PyStr, PyStrRef, PyType, PyTypeRef, PyWeak};
use crate::common::lock::LazyLock;
use crate::{
    AsObject, Context, Py, PyObject, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine,
    atomic_func,
    class::PyClassImpl,
    function::{FuncArgs, OptionalArg, PyComparisonValue, PySetterValue},
    protocol::{PyIter, PyIterReturn, PyMappingMethods, PyNumberMethods, PySequenceMethods},
    stdlib::builtins::reversed,
    types::{
        AsMapping, AsNumber, AsSequence, Callable, Comparable, Constructor, GetAttr, IterNext,
        Iterable, PyComparisonOp, Representable, SetAttr,
    },
};

#[pyclass(module = false, name = "weakproxy", unhashable = true, traverse)]
#[derive(Debug)]
pub struct PyWeakProxy {
    weak: PyRef<PyWeak>,
}

impl PyPayload for PyWeakProxy {
    #[inline]
    fn class(ctx: &Context) -> &'static Py<PyType> {
        ctx.types.weakproxy_type
    }
}

#[derive(FromArgs)]
pub struct WeakProxyNewArgs {
    #[pyarg(positional)]
    referent: PyObjectRef,
    #[pyarg(positional, optional)]
    callback: OptionalArg<PyObjectRef>,
}

impl PyWeakProxy {
    pub fn new(
        referent: PyObjectRef,
        callback: Option<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<Self> {
        Ok(Self {
            weak: make_weak_ref(referent, callback, vm)?,
        })
    }

    fn try_upgrade(&self, vm: &VirtualMachine) -> PyResult {
        proxy_try_upgrade(&self.weak, vm)
    }
}

crate::common::static_cell! {
    static WEAK_SUBCLASS: PyTypeRef;
}

#[pyclass(with(
    GetAttr,
    SetAttr,
    Constructor,
    Comparable,
    AsNumber,
    AsSequence,
    AsMapping,
    Representable,
    IterNext
))]
impl PyWeakProxy {
    #[pymethod]
    fn __str__(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<PyStrRef> {
        zelf.try_upgrade(vm)?.str(vm)
    }

    #[pymethod]
    fn __bytes__(&self, vm: &VirtualMachine) -> PyResult {
        self.try_upgrade(vm)?.bytes(vm)
    }

    #[pymethod]
    fn __reversed__(&self, vm: &VirtualMachine) -> PyResult {
        reversed(self.try_upgrade(vm)?, vm)
    }
}

fn new_reference_error(vm: &VirtualMachine) -> PyRef<super::PyBaseException> {
    vm.new_exception_msg(
        vm.ctx.exceptions.reference_error.to_owned(),
        "weakly-referenced object no longer exists".into(),
    )
}

fn proxy_try_upgrade(weak: &PyRef<PyWeak>, vm: &VirtualMachine) -> PyResult {
    weak.upgrade().ok_or_else(|| new_reference_error(vm))
}

/// If `obj` is a proxy, upgrade it to the referent. Otherwise return `None`.
fn proxy_upgrade_opt(obj: &PyObject, vm: &VirtualMachine) -> PyResult<Option<PyObjectRef>> {
    if let Some(proxy) = obj.downcast_ref::<PyWeakProxy>() {
        Ok(Some(proxy_try_upgrade(&proxy.weak, vm)?))
    } else if let Some(proxy) = obj.downcast_ref::<PyWeakCallableProxy>() {
        Ok(Some(proxy_try_upgrade(&proxy.weak, vm)?))
    } else {
        Ok(None)
    }
}

fn proxy_repr(id: usize, weak: &PyRef<PyWeak>) -> String {
    if let Some(obj) = weak.upgrade() {
        format!(
            "<weakproxy at {:#x}; to '{}' at {:#x}>",
            id,
            obj.class().name(),
            obj.get_id(),
        )
    } else {
        format!("<weakproxy at {id:#x}; dead>")
    }
}

fn make_weak_ref(
    referent: PyObjectRef,
    callback: Option<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<PyRef<PyWeak>> {
    let weak_cls = WEAK_SUBCLASS.get_or_init(|| {
        vm.ctx.new_class(
            None,
            "__weakproxy",
            vm.ctx.types.weakref_type.to_owned(),
            super::PyWeak::make_slots(),
        )
    });
    referent.downgrade_with_typ(callback, weak_cls.clone(), vm)
}

fn proxy_unary_op(
    obj: &PyObject,
    vm: &VirtualMachine,
    op: fn(&VirtualMachine, &PyObject) -> PyResult,
) -> PyResult {
    let upgraded = proxy_upgrade_opt(obj, vm)?.unwrap_or_else(|| obj.to_owned());
    op(vm, &upgraded)
}

macro_rules! proxy_unary_slot {
    ($vm_method:ident) => {
        Some(|number, vm| proxy_unary_op(number.obj, vm, |vm, obj| vm.$vm_method(obj)))
    };
}

fn proxy_binary_op(
    a: &PyObject,
    b: &PyObject,
    vm: &VirtualMachine,
    op: fn(&VirtualMachine, &PyObject, &PyObject) -> PyResult,
) -> PyResult {
    let a_up = proxy_upgrade_opt(a, vm)?;
    let b_up = proxy_upgrade_opt(b, vm)?;
    let a_ref = a_up.as_deref().unwrap_or(a);
    let b_ref = b_up.as_deref().unwrap_or(b);
    op(vm, a_ref, b_ref)
}

macro_rules! proxy_binary_slot {
    ($vm_method:ident) => {
        Some(|a, b, vm| proxy_binary_op(a, b, vm, |vm, a, b| vm.$vm_method(a, b)))
    };
}

fn proxy_ternary_op(
    a: &PyObject,
    b: &PyObject,
    c: &PyObject,
    vm: &VirtualMachine,
    op: fn(&VirtualMachine, &PyObject, &PyObject, &PyObject) -> PyResult,
) -> PyResult {
    let a_up = proxy_upgrade_opt(a, vm)?;
    let b_up = proxy_upgrade_opt(b, vm)?;
    let c_up = proxy_upgrade_opt(c, vm)?;
    let a_ref = a_up.as_deref().unwrap_or(a);
    let b_ref = b_up.as_deref().unwrap_or(b);
    let c_ref = c_up.as_deref().unwrap_or(c);
    op(vm, a_ref, b_ref, c_ref)
}

macro_rules! proxy_ternary_slot {
    ($vm_method:ident) => {
        Some(|a, b, c, vm| proxy_ternary_op(a, b, c, vm, |vm, a, b, c| vm.$vm_method(a, b, c)))
    };
}

macro_rules! proxy_number_methods {
    ($proxy_type:ty) => {{
        PyNumberMethods {
            boolean: Some(|number, vm| {
                let zelf = number.obj.downcast_ref::<$proxy_type>().unwrap();
                proxy_try_upgrade(&zelf.weak, vm)?.is_true(vm)
            }),
            int: Some(|number, vm| {
                let obj = proxy_upgrade_opt(number.obj, vm)?.unwrap_or_else(|| number.obj.to_owned());
                obj.try_int(vm).map(Into::into)
            }),
            float: Some(|number, vm| {
                let obj = proxy_upgrade_opt(number.obj, vm)?.unwrap_or_else(|| number.obj.to_owned());
                obj.try_float(vm).map(Into::into)
            }),
            index: Some(|number, vm| {
                let obj = proxy_upgrade_opt(number.obj, vm)?.unwrap_or_else(|| number.obj.to_owned());
                obj.try_index(vm).map(Into::into)
            }),
            negative: proxy_unary_slot!(_neg),
            positive: proxy_unary_slot!(_pos),
            absolute: proxy_unary_slot!(_abs),
            invert: proxy_unary_slot!(_invert),
            add: proxy_binary_slot!(_add),
            subtract: proxy_binary_slot!(_sub),
            multiply: proxy_binary_slot!(_mul),
            remainder: proxy_binary_slot!(_mod),
            divmod: proxy_binary_slot!(_divmod),
            lshift: proxy_binary_slot!(_lshift),
            rshift: proxy_binary_slot!(_rshift),
            and: proxy_binary_slot!(_and),
            xor: proxy_binary_slot!(_xor),
            or: proxy_binary_slot!(_or),
            floor_divide: proxy_binary_slot!(_floordiv),
            true_divide: proxy_binary_slot!(_truediv),
            matrix_multiply: proxy_binary_slot!(_matmul),
            inplace_add: proxy_binary_slot!(_iadd),
            inplace_subtract: proxy_binary_slot!(_isub),
            inplace_multiply: proxy_binary_slot!(_imul),
            inplace_remainder: proxy_binary_slot!(_imod),
            inplace_lshift: proxy_binary_slot!(_ilshift),
            inplace_rshift: proxy_binary_slot!(_irshift),
            inplace_and: proxy_binary_slot!(_iand),
            inplace_xor: proxy_binary_slot!(_ixor),
            inplace_or: proxy_binary_slot!(_ior),
            inplace_floor_divide: proxy_binary_slot!(_ifloordiv),
            inplace_true_divide: proxy_binary_slot!(_itruediv),
            inplace_matrix_multiply: proxy_binary_slot!(_imatmul),
            power: proxy_ternary_slot!(_pow),
            inplace_power: proxy_ternary_slot!(_ipow),
        }
    }};
}

macro_rules! impl_proxy_traits {
    ($proxy_type:ty) => {
        impl Constructor for $proxy_type {
            type Args = WeakProxyNewArgs;

            fn py_new(
                _cls: &Py<PyType>,
                Self::Args { referent, callback }: Self::Args,
                vm: &VirtualMachine,
            ) -> PyResult<Self> {
                Self::new(referent, callback.into_option(), vm)
            }
        }

        impl Iterable for $proxy_type {
            fn iter(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult {
                let obj = zelf.try_upgrade(vm)?;
                Ok(obj.get_iter(vm)?.into())
            }
        }

        impl IterNext for $proxy_type {
            fn next(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<PyIterReturn> {
                let obj = zelf.try_upgrade(vm)?;
                if !PyIter::check(&obj) {
                    return Err(vm.new_type_error(format!(
                        "Weakref proxy referenced a non-iterator '{}' object",
                        obj.class().name()
                    )));
                }
                PyIter::new(obj).next(vm)
            }
        }

        impl GetAttr for $proxy_type {
            fn getattro(zelf: &Py<Self>, name: &Py<PyStr>, vm: &VirtualMachine) -> PyResult {
                let obj = zelf.try_upgrade(vm)?;
                obj.get_attr(name, vm)
            }
        }

        impl SetAttr for $proxy_type {
            fn setattro(
                zelf: &Py<Self>,
                attr_name: &Py<PyStr>,
                value: PySetterValue,
                vm: &VirtualMachine,
            ) -> PyResult<()> {
                let obj = zelf.try_upgrade(vm)?;
                obj.call_set_attr(vm, attr_name, value)
            }
        }

        impl AsNumber for $proxy_type {
            fn as_number() -> &'static PyNumberMethods {
                static AS_NUMBER: LazyLock<PyNumberMethods> =
                    LazyLock::new(|| proxy_number_methods!($proxy_type));
                &AS_NUMBER
            }
        }

        impl Comparable for $proxy_type {
            fn cmp(
                zelf: &Py<Self>,
                other: &PyObject,
                op: PyComparisonOp,
                vm: &VirtualMachine,
            ) -> PyResult<PyComparisonValue> {
                let obj = zelf.try_upgrade(vm)?;
                let other_up = proxy_upgrade_opt(other, vm)?;
                let other_ref = other_up.as_deref().unwrap_or(other);
                Ok(PyComparisonValue::Implemented(
                    obj.rich_compare_bool(other_ref, op, vm)?,
                ))
            }
        }

        impl AsSequence for $proxy_type {
            fn as_sequence() -> &'static PySequenceMethods {
                static AS_SEQUENCE: LazyLock<PySequenceMethods> =
                    LazyLock::new(|| PySequenceMethods {
                        length: atomic_func!(|seq, vm| {
                            let zelf = <$proxy_type>::sequence_downcast(seq);
                            zelf.try_upgrade(vm)?.length(vm)
                        }),
                        contains: atomic_func!(|seq, needle, vm| {
                            let zelf = <$proxy_type>::sequence_downcast(seq);
                            zelf.try_upgrade(vm)?
                                .sequence_unchecked()
                                .contains(needle, vm)
                        }),
                        ..PySequenceMethods::NOT_IMPLEMENTED
                    });
                &AS_SEQUENCE
            }
        }

        impl AsMapping for $proxy_type {
            fn as_mapping() -> &'static PyMappingMethods {
                static AS_MAPPING: PyMappingMethods = PyMappingMethods {
                    length: atomic_func!(|mapping, vm| {
                        let zelf = <$proxy_type>::mapping_downcast(mapping);
                        zelf.try_upgrade(vm)?.length(vm)
                    }),
                    subscript: atomic_func!(|mapping, needle, vm| {
                        let zelf = <$proxy_type>::mapping_downcast(mapping);
                        zelf.try_upgrade(vm)?.get_item(needle, vm)
                    }),
                    ass_subscript: atomic_func!(|mapping, needle, value, vm| {
                        let obj = <$proxy_type>::mapping_downcast(mapping).try_upgrade(vm)?;
                        if let Some(value) = value {
                            obj.set_item(needle, value, vm)
                        } else {
                            obj.del_item(needle, vm)
                        }
                    }),
                };
                &AS_MAPPING
            }
        }

        impl Representable for $proxy_type {
            #[inline]
            fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
                Ok(proxy_repr(zelf.get_id(), &zelf.weak))
            }
        }
    };
}

impl_proxy_traits!(PyWeakProxy);

#[pyclass(module = false, name = "weakcallableproxy", unhashable = true, traverse)]
#[derive(Debug)]
pub struct PyWeakCallableProxy {
    weak: PyRef<PyWeak>,
}

impl PyPayload for PyWeakCallableProxy {
    #[inline]
    fn class(ctx: &Context) -> &'static Py<PyType> {
        ctx.types.weakcallableproxy_type
    }
}

impl PyWeakCallableProxy {
    pub fn new(
        referent: PyObjectRef,
        callback: Option<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<Self> {
        Ok(Self {
            weak: make_weak_ref(referent, callback, vm)?,
        })
    }

    fn try_upgrade(&self, vm: &VirtualMachine) -> PyResult {
        proxy_try_upgrade(&self.weak, vm)
    }
}

#[pyclass(with(
    GetAttr,
    SetAttr,
    Constructor,
    Comparable,
    Callable,
    AsNumber,
    AsSequence,
    AsMapping,
    Representable,
    IterNext
))]
impl PyWeakCallableProxy {
    #[pymethod]
    fn __str__(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<PyStrRef> {
        zelf.try_upgrade(vm)?.str(vm)
    }

    #[pymethod]
    fn __bytes__(&self, vm: &VirtualMachine) -> PyResult {
        self.try_upgrade(vm)?.bytes(vm)
    }

    #[pymethod]
    fn __reversed__(&self, vm: &VirtualMachine) -> PyResult {
        reversed(self.try_upgrade(vm)?, vm)
    }
}

impl Callable for PyWeakCallableProxy {
    type Args = FuncArgs;

    fn call(zelf: &Py<Self>, args: Self::Args, vm: &VirtualMachine) -> PyResult {
        zelf.try_upgrade(vm)?.call(args, vm)
    }
}

impl_proxy_traits!(PyWeakCallableProxy);

pub fn init(context: &'static Context) {
    PyWeakProxy::extend_class(context, context.types.weakproxy_type);
    PyWeakCallableProxy::extend_class(context, context.types.weakcallableproxy_type);
}
