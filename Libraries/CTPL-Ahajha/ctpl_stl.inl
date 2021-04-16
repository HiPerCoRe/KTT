/*********************************************************
*
*  Copyright (C) 2014 by Vitaliy Vitsentiy
*  Modifications: Copyright (C) 2021 by Alex Trotta
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*********************************************************/

#include "ctpl_stl.h"

namespace ctpl
{

namespace detail
{

template <typename T>
bool atomic_queue<T>::push(T&& value)
{
	std::lock_guard lock(this->mut);
	this->q.push(std::forward<T>(value));
	return true;
}

template <typename T>
bool atomic_queue<T>::pop(T & value)
{
	std::lock_guard lock(this->mut);
	if (this->q.empty())
		return false;
	value = std::move(this->q.front());
	this->q.pop();
	return true;
}

template <typename T>
bool atomic_queue<T>::empty() const
{
	std::lock_guard lock(this->mut);
	return this->q.empty();
}

template <typename T>
void atomic_queue<T>::clear()
{
	std::lock_guard lock(this->mut);
	while (!this->q.empty()) this->q.pop();
}

} // namespace detail

template<typename F, typename... Args>
std::future<std::invoke_result_t<F,Args...>> thread_pool::push(F && f, Args&&... args)
{
	// std::packaged_task is used to get a future out of the function call.
	auto pck = std::make_shared<std::packaged_task<std::invoke_result_t<F,Args...>()>>(
		// This has been tested to ensure perfect forwarding still occurs with
		// the parameters captured by reference.
		[&f,&args...]
		{
			if constexpr (sizeof...(args) == 0)
			{
				// Only need to forward the function.
				return std::forward<F>(f);
			}
			else
			{
				// std::forward is used to ensure perfect
				//     forwarding of rvalues where necessary.
				// std::bind is used to make a parameterless function
				//     that simulates calling f with its respective arguments.
				return std::bind(std::forward<F>(f), std::forward<Args>(args)...);
			}
		}()
	);
	
	// Lastly, create a function that wraps the packaged
	// task into a signature of void().
	this->tasks.push(std::make_unique<std::function<void()>>(
		[pck]{ (*pck)(); })
	);
	
	// Notify one waiting thread so it can wake up and take this new task.
	std::lock_guard lock(this->signal_mut);
	this->signal.notify_one();
	
	// Return the future, the user is now responsible
	// for the return value of the task.
	return pck->get_future();
}

} // namespace ctpl
